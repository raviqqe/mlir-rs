use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;

pub fn generate(identifiers: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for identifier in identifiers {
        let mut name = identifier.to_string();

        if let Some(other) = name.strip_prefix("mlirCreate") {
            name = other.into();
        }

        if let Some(other) = name.strip_suffix("Pass") {
            name = other.into();
        }

        let name = Ident::new(&name.to_case(Case::Snake), identifier.span());

        stream.extend(TokenStream::from(quote! {
            pub fn #name() -> melior::pass::Pass {
                melior::pass::Pass::_macro_from_raw_fn(mlir_sys::#identifier)
            }
        }));
    }

    Ok(stream.into())
}
