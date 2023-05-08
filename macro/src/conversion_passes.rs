use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;

pub fn generate(identifiers: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for identifier in identifiers {
        let mut name = identifier.to_string();

        if let Some(other) = name.strip_prefix("mlirCreateConversion") {
            name = other.into();
        }

        if let Some(other) = name.strip_suffix("ConversionPass") {
            name = other.into();
        }

        let function_name = Ident::new(&name.to_case(Case::Snake), identifier.span());
        let document = format!(" Creates a pass of `{}`.", name);

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #function_name() -> crate::pass::Pass {
                crate::pass::Pass::__private_from_raw_fn(mlir_sys::#identifier)
            }
        }));
    }

    Ok(stream.into())
}
