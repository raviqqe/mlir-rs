use crate::dialect::{
    error::Error,
    operation::{OperationFieldLike, Region},
};
use proc_macro2::TokenStream;
use quote::quote;

pub fn generate_region_accessor(region: &Region) -> Result<TokenStream, Error> {
    let idenifier = &region.singular_identifier();
    let return_type = &region.return_type();
    let body = generate_getter(region);

    Ok(quote! {
        #[allow(clippy::needless_question_mark)]
        pub fn #idenifier(&self, context: &'c ::melior::Context) -> #return_type {
            #body
        }
    })
}

fn generate_getter(region: &Region) -> TokenStream {
    let index = region.sequence_info().index;

    if region.is_variadic() {
        // Only the last region can be variadic
        quote! {
            self.operation.regions().skip(#index)
        }
    } else {
        quote! {
            self.operation.region(#index)
        }
    }
}
