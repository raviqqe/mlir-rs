use crate::dialect::operation::{OperationField, Region};
use proc_macro2::TokenStream;
use quote::quote;

pub fn generate_region_accessor(region: &Region, index: usize) -> TokenStream {
    let identifier = &region.singular_identifier();
    let return_type = &region.return_type();
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

    quote! {
        pub fn #identifier(&self) -> #return_type {
            #body
        }
    }
}
