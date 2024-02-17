use crate::dialect::{
    error::Error,
    operation::{FieldKind, OperationField, SequenceInfo, VariadicKind},
};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::Ident;

pub fn generate_accessor(field: &OperationField) -> Result<TokenStream, Error> {
    let ident = &field.sanitized_name;
    let return_type = &field.kind.return_type();
    let body = generate_getter(field);

    Ok(quote! {
        #[allow(clippy::needless_question_mark)]
        pub fn #ident(&self, context: &'c ::melior::Context) -> #return_type {
            #body
        }
    })
}

fn generate_getter(field: &OperationField) -> TokenStream {
    match &field.kind {
        FieldKind::Element {
            constraint,
            sequence_info: SequenceInfo { index, len },
            variadic_kind,
        } => {
            let kind_ident = Ident::new("operand", Span::call_site());
            let plural = Ident::new("operands", Span::call_site());
            let count = Ident::new("operand_count", Span::call_site());
            let error_variant = quote!(OperandNotFound);
            let name = field.name;

            match variadic_kind {
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
                    let get_elements = if constraint.is_unfixed() {
                        quote! {
                            self.operation.#plural().skip(start).take(group_len)
                        }
                    } else {
                        quote! {
                            self.operation.#kind_ident(start)
                        }
                    };

                    quote! {
                        let total_var_len = self.operation.#count() - #unfixed_count + 1;
                        let group_len = total_var_len / #unfixed_count;
                        let start = #preceding_simple_count + #preceding_variadic_count * group_len;

                        #get_elements
                    }
                }
                VariadicKind::AttributeSized => {
                    let get_elements = if !constraint.is_unfixed() {
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

                    quote! {
                        let attribute =
                            ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
                                self.operation
                                .attribute("operand_segment_sizes")?
                            )?;
                        let start = (0..#index)
                            .map(|index| attribute.element(index))
                            .collect::<Result<Vec<_>, _>>()?
                            .into_iter()
                            .sum::<i32>() as usize;
                        let group_len = attribute.element(#index)? as usize;

                        #get_elements
                    }
                }
            }
        }
    }
}
