use crate::dialect::operation::{OperationElement, OperationFieldLike, VariadicKind};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub fn generate_element_getter(
    field: &(impl OperationFieldLike + OperationElement),
    singular_kind: &str,
    plural_kind: &str,
    index: usize,
    length: usize,
) -> TokenStream {
    let kind_singular_identifier = Ident::new(singular_kind, Span::call_site());
    let kind_plural_identifier = Ident::new(plural_kind, Span::call_site());
    let count = Ident::new(&format!("{singular_kind}_count"), Span::call_site());
    let error_variant = quote!(ResultNotFound);
    let name = field.name();

    match field.variadic_kind() {
        VariadicKind::Simple { unfixed_seen } => {
            if field.is_optional() {
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
            } else if field.is_variadic() {
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
            let get_elements = if field.is_unfixed() {
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
            let segment_size_attribute = format!("{singular_kind}_segment_sizes");
            let get_elements = if !field.is_unfixed() {
                quote! {
                    self.operation.#kind_singular_identifier(start)
                }
            } else if field.is_optional() {
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
                        .attribute(#segment_size_attribute)?
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
