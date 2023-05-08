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

        if let Some(other) = name.strip_prefix("Convert") {
            name = other.into();
        }

        if let Some(other) = name.strip_suffix("ConversionPass") {
            name = other.into();
        }

        let name = name.to_case(Case::Snake);
        let dialects = name.split("_to_").collect::<Vec<_>>();
        let function_name = Ident::new(&name, identifier.span());
        let document = format!(
            " Converts `{}` dialect to `{}` dialect.",
            map_dialect_name(&dialects[0]),
            map_dialect_name(&dialects[1]),
        );

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #function_name() -> crate::pass::Pass {
                crate::pass::Pass::_macro_from_raw_fn(mlir_sys::#identifier)
            }
        }));
    }

    Ok(stream.into())
}

fn map_dialect_name(name: &str) -> &str {
    match name {
        "control_flow" => "cf",
        name => name,
    }
}
