use super::{
    super::{error::Error, utility::sanitize_snake_case_name},
    ElementKind, FieldKind, OperationField, SequenceInfo, VariadicKind,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

impl<'a> OperationField<'a> {
    fn getter_impl(&self) -> Result<Option<TokenStream>, Error> {
        Ok(match &self.kind {
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
                let name = self.name;

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
                let name = &self.name;

                Some(if constraint.is_unit()? {
                    quote! { self.operation.attribute(#name).is_some() }
                } else {
                    quote! { Ok(self.operation.attribute(#name)?.try_into()?) }
                })
            }
        })
    }

    fn remover_impl(&self) -> Result<Option<TokenStream>, Error> {
        Ok(match &self.kind {
            FieldKind::Attribute { constraint } => {
                let name = &self.name;

                if constraint.is_unit()? || constraint.is_optional()? {
                    Some(quote! {
                      self.operation.remove_attribute(#name)
                    })
                } else {
                    None
                }
            }
            _ => None,
        })
    }

    fn setter_impl(&self) -> Result<Option<TokenStream>, Error> {
        let FieldKind::Attribute { constraint } = &self.kind else {
            return Ok(None);
        };
        let name = &self.name;

        Ok(Some(if constraint.is_unit()? {
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
        }))
    }

    pub fn accessors(&self) -> Result<TokenStream, Error> {
        let setter = {
            let ident = sanitize_snake_case_name(&format!("set_{}", self.name))?;

            if let Some(body) = self.setter_impl()? {
                let parameter_type = &self.kind.parameter_type()?;

                quote! {
                    pub fn #ident(&mut self, context: &'c ::melior::Context, value: #parameter_type) {
                        #body
                    }
                }
            } else {
                quote!()
            }
        };
        let remover = {
            let ident = sanitize_snake_case_name(&format!("remove_{}", self.name))?;
            self.remover_impl()?.map(|body| {
                quote! {
                    pub fn #ident(&mut self, context: &'c ::melior::Context) -> Result<(), ::melior::Error> {
                        #body
                    }
                }
            })
        };
        let getter = {
            let ident = &self.sanitized_name;
            let return_type = &self.kind.return_type()?;
            self.getter_impl()?.map(|body| {
                quote! {
                    pub fn #ident(&self, context: &'c ::melior::Context) -> #return_type {
                        #body
                    }
                }
            })
        };

        Ok(quote! {
            #getter
            #setter
            #remover
        })
    }
}
