use super::element_accessor::generate_element_getter;
use crate::dialect::{error::Error, operation::Operand};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

pub fn generate_operand_accessor(
    field: &Operand,
    index: usize,
    length: usize,
) -> Result<TokenStream, Error> {
    let ident = &field.sanitized_name;
    let return_type = &field.kind.return_type();
    let body = generate_element_getter(
        field,
        "operand",
        "operands",
        &Ident::new("OperandNotFound", Span::call_site()),
        index,
        length,
    );

    Ok(quote! {
        #[allow(clippy::needless_question_mark)]
        pub fn #ident(&self, context: &'c ::melior::Context) -> #return_type {
            #body
        }
    })
}
