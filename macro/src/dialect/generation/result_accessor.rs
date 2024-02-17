use crate::dialect::{
    error::Error,
    operation::{OperationFieldLike, OperationResult, VariadicKind},
};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub fn generate_result_accessor(
    result: &OperationResult,
    index: usize,
    length: usize,
) -> Result<TokenStream, Error> {
    let identifier = result.singular_identifier();
    let return_type = result.return_type();
    let body = generate_getter(result, index, length);

    Ok(quote! {
        #[allow(clippy::needless_question_mark)]
        pub fn #identifier(&self, context: &'c ::melior::Context) -> #return_type {
            #body
        }
    })
}

// TODO Share this logic with `Operand`.
fn generate_getter(result: &OperationResult, index: usize, length: usize) -> TokenStream {
    let kind_singular_identifier = Ident::new("result", Span::call_site());
    let kind_plural_identifier = Ident::new("results", Span::call_site());
    let count = Ident::new("result_count", Span::call_site());
    let error_variant = quote!(ResultNotFound);
    let name = result.name();

    match result.variadic_kind() {
        VariadicKind::Simple { unfixed_seen } => {
            if result.is_optional() {
                // Optional element, and some singular elements.
                // Only present if the amount of groups is at least the number of
                // elements.
                quote! {
                    if self.operation.#count() < #length {
                        Err(::melior::Error::#error_variant(#name))
                    } else {
                        self.operation.#kind_singular_identifier(#index)
                    }
                }
            } else if result.is_variadic() {
                // A unfixed group
                // Length computed by subtracting the amount of other
                // singular elements from the number of elements.
                quote! {
                    let group_length = self.operation.#count() - #length + 1;
                    self.operation.#kind_plural_identifier().skip(#index).take(group_length)
                }
            } else if *unfixed_seen {
                // Single element after unfixed group
                // Compute the length of that variable group and take the next element
                quote! {
                    let group_length = self.operation.#count() - #length + 1;
                    self.operation.#kind_singular_identifier(#index + group_length - 1)
                }
            } else {
                // All elements so far are singular
                quote! {
                    self.operation.#kind_singular_identifier(#index)
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
            let get_elements = if result.is_unfixed() {
                quote! {
                    self.operation.#kind_plural_identifier().skip(start).take(group_len)
                }
            } else {
                quote! {
                    self.operation.#kind_singular_identifier(start)
                }
            };

            quote! { #compute_start_length #get_elements }
        }
        VariadicKind::AttributeSized => {
            let get_elements = if !result.is_unfixed() {
                quote! {
                    self.operation.#kind_singular_identifier(start)
                }
            } else if result.is_optional() {
                quote! {
                    if group_len == 0 {
                        Err(::melior::Error::#error_variant(#name))
                    } else {
                        self.operation.#kind_singular_identifier(start)
                    }
                }
            } else {
                quote! {
                    Ok(self.operation.#kind_plural_identifier().skip(start).take(group_len))
                }
            };

            quote! {
                let attribute =
                    ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
                        self.operation
                        .attribute("result_segment_sizes")?
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
