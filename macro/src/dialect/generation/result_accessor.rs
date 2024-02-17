use super::element_accessor::generate_element_getter;
use crate::dialect::{
    error::Error,
    operation::{OperationFieldLike, OperationResult},
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

pub fn generate_result_accessor(
    result: &OperationResult,
    index: usize,
    length: usize,
) -> Result<TokenStream, Error> {
    let identifier = result.singular_identifier();
    let return_type = result.return_type();
    let body = generate_element_getter(
        result,
        "result",
        "results",
        &Ident::new("ResultNotFound", Span::call_site()),
        index,
        length,
    );

    Ok(quote! {
        #[allow(clippy::needless_question_mark)]
        pub fn #identifier(&self, context: &'c ::melior::Context) -> #return_type {
            #body
        }
    })
}
