use super::operation_field::OperationFieldV2;
use super::{
    super::{error::Error, utility::sanitize_snake_case_name},
    ElementKind, FieldKind, OperationField, SequenceInfo, VariadicKind,
};
use crate::dialect::types::AttributeConstraint;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_quote, Ident};

#[derive(Debug)]
pub struct Attribute<'a> {
    name: &'a str,
    sanitized_name: Ident,
    constraint: AttributeConstraint<'a>,
}

impl<'a> Attribute<'a> {
    pub fn new(name: &'a str, constraint: AttributeConstraint<'a>) -> Result<Self, Error> {
        Ok(Self {
            name,
            sanitized_name: sanitize_snake_case_name(name)?,
            constraint,
        })
    }

    pub fn constraint(&self) -> &AttributeConstraint {
        &self.constraint
    }
}

impl OperationFieldV2 for Attribute<'_> {
    fn name(&self) -> &str {
        &self.name
    }

    fn sanitized_name(&self) -> &Ident {
        &self.sanitized_name
    }
}

pub fn generate_accessors(attribute: &Attribute) -> Result<TokenStream, Error> {
    let getter = {
        let ident = &field.sanitized_name;
        let return_type = &field.kind.return_type();
        generate_getter(field).map(|body| {
            quote! {
                #[allow(clippy::needless_question_mark)]
                pub fn #ident(&self, context: &'c ::melior::Context) -> #return_type {
                    #body
                }
            }
        })
    };
    let setter = generate_setter(attribute)?;
    let remover = generate_remover(attribute)?;

    Ok(quote! {
        #getter
        #setter
        #remover
    })
}

fn generate_getter(field: &OperationField) -> Option<TokenStream> {
    match &field.kind {
        FieldKind::Element {
            kind,
            constraint,
            sequence_info: SequenceInfo { index, len },
            variadic_kind,
        } => {
            let kind_ident = format_ident!("{}", kind.as_str());
            let plural = format_ident!("{}s", kind.as_str());
            let count = format_ident!("{}_count", kind.as_str());
            let error_variant = match kind {
                ElementKind::Operand => quote!(OperandNotFound),
                ElementKind::Result => quote!(ResultNotFound),
            };
            let name = field.name;

            Some(match variadic_kind {
                VariadicKind::Simple { unfixed_seen } => {
                    if constraint.is_optional() {
                        // Optional element, and some singular elements.
                        // Only present if the amount of groups is at least the number of
                        // elements.
                        quote! {
                            if self.operation.#count() < #len {
                                Err(::melior::Error::#error_variant(#name))
                            } else {
                                self.operation.#kind_ident(#index)
                            }
                        }
                    } else if constraint.is_variadic() {
                        // A unfixed group
                        // Length computed by subtracting the amount of other
                        // singular elements from the number of elements.
                        quote! {
                            let group_length = self.operation.#count() - #len + 1;
                            self.operation.#plural().skip(#index).take(group_length)
                        }
                    } else if *unfixed_seen {
                        // Single element after unfixed group
                        // Compute the length of that variable group and take the next element
                        quote! {
                            let group_length = self.operation.#count() - #len + 1;
                            self.operation.#kind_ident(#index + group_length - 1)
                        }
                    } else {
                        // All elements so far are singular
                        quote! {
                            self.operation.#kind_ident(#index)
                        }
                    }
                }
                VariadicKind::SameSize {
                    unfixed_count,
                    preceding_simple_count,
                    preceding_variadic_count,
                } => {
                    let compute_start_length = quote! {
                        let total_var_len = self.operation.#count() - #unfixed_count + 1;
                        let group_len = total_var_len / #unfixed_count;
                        let start = #preceding_simple_count + #preceding_variadic_count * group_len;
                    };
                    let get_elements = if constraint.has_unfixed() {
                        quote! {
                            self.operation.#plural().skip(start).take(group_len)
                        }
                    } else {
                        quote! {
                            self.operation.#kind_ident(start)
                        }
                    };

                    quote! { #compute_start_length #get_elements }
                }
                VariadicKind::AttributeSized => {
                    let attribute_name = format!("{}_segment_sizes", kind.as_str());
                    let compute_start_length = quote! {
                        let attribute =
                            ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
                                self.operation
                                .attribute(#attribute_name)?
                            )?;
                        let start = (0..#index)
                            .map(|index| attribute.element(index))
                            .collect::<Result<Vec<_>, _>>()?
                            .into_iter()
                            .sum::<i32>() as usize;
                        let group_len = attribute.element(#index)? as usize;
                    };
                    let get_elements = if !constraint.has_unfixed() {
                        quote! {
                            self.operation.#kind_ident(start)
                        }
                    } else if constraint.is_optional() {
                        quote! {
                            if group_len == 0 {
                                Err(::melior::Error::#error_variant(#name))
                            } else {
                                self.operation.#kind_ident(start)
                            }
                        }
                    } else {
                        quote! {
                            Ok(self.operation.#plural().skip(start).take(group_len))
                        }
                    };

                    quote! { #compute_start_length #get_elements }
                }
            })
        }
        FieldKind::Successor {
            constraint,
            sequence_info: SequenceInfo { index, .. },
        } => {
            Some(if constraint.is_variadic() {
                // Only the last successor can be variadic
                quote! {
                    self.operation.successors().skip(#index)
                }
            } else {
                quote! {
                    self.operation.successor(#index)
                }
            })
        }
        FieldKind::Region {
            constraint,
            sequence_info: SequenceInfo { index, .. },
        } => {
            Some(if constraint.is_variadic() {
                // Only the last region can be variadic
                quote! {
                    self.operation.regions().skip(#index)
                }
            } else {
                quote! {
                    self.operation.region(#index)
                }
            })
        }
        FieldKind::Attribute { constraint } => {
            let name = &field.name;

            Some(if constraint.is_unit() {
                quote! { self.operation.attribute(#name).is_some() }
            } else {
                // TODO Handle returning `melior::Attribute`.
                quote! { Ok(self.operation.attribute(#name)?.try_into()?) }
            })
        }
    }
}

fn generate_setter(attribute: &Attribute) -> Result<TokenStream, Error> {
    let name = attribute.name();

    let body = if attribute.constraint().is_unit() {
        quote! {
            if value {
                self.operation.set_attribute(#name, Attribute::unit(&self.operation.context()));
            } else {
                self.operation.remove_attribute(#name)
            }
        }
    } else {
        quote! {
            self.operation.set_attribute(#name, &value.into());
        }
    };

    let ident = sanitize_snake_case_name(&format!("set_{}", attribute.name()))?;
    let r#type: Type = if attribute.constraint().is_unit() {
        parse_quote!(bool)
    } else {
        let r#type = attribute.constraint().storage_type();
        parse_quote!(#r#type<'c>)
    };

    Ok(quote! {
        pub fn #ident(&mut self, context: &'c ::melior::Context, value: #r#type) {
            #body
        }
    })
}

fn generate_remover(attribute: &Attribute) -> Result<Option<TokenStream>, Error> {
    let constrait = attribute.constraint();
    let ident = sanitize_snake_case_name(&format!("remove_{}", attribute.name()))?;

    Ok(if constrait.is_unit() || constrait.is_optional() {
        let name = attribute.name();

        Some(quote! {
            pub fn #ident(&mut self, context: &'c ::melior::Context) -> Result<(), ::melior::Error> {
                self.operation.remove_attribute(#name)
            }
        })
    } else {
        None
    })
}
