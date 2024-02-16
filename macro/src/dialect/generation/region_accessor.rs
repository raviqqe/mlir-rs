use crate::dialect::{
    error::Error,
    operation::{OperationFieldLike, Region},
};
use proc_macro2::TokenStream;
use quote::quote;

pub fn generate_region_accessor(region: &Region) -> Result<TokenStream, Error> {
    let identifier = &region.singular_identifier();
    let return_type = &region.return_type();
    let index = region.sequence_info().index;
    let body = if region.is_variadic() {
        // Only the last region can be variadic.
        quote! {
            self.operation.regions().skip(#index)
        }
    } else {
        quote! {
            self.operation.region(#index)
        }
    };

    Ok(quote! {
        #[allow(clippy::needless_question_mark)]
        pub fn #identifier(&self, context: &'c ::melior::Context) -> #return_type {
            #body
        }
    })
}
