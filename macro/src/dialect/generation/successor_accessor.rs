use crate::dialect::{
    error::Error,
    operation::{OperationFieldLike, Successor},
};
use proc_macro2::TokenStream;
use quote::quote;

pub fn generate_successor_accessor(
    index: usize,
    successor: &Successor,
) -> Result<TokenStream, Error> {
    let identifier = successor.singular_identifier();
    let return_type = successor.return_type();
    let body = if successor.is_variadic() {
        // Only the last successor can be variadic.
        quote! {
            self.operation.successors().skip(#index)
        }
    } else {
        quote! {
            self.operation.successor(#index)
        }
    };

    Ok(quote! {
        #[allow(clippy::needless_question_mark)]
        pub fn #identifier(&self, context: &'c ::melior::Context) -> #return_type {
            #body
        }
    })
}
