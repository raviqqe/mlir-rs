use crate::passes;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use std::error::Error;

pub fn generate(identifiers: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    passes::generate(identifiers, extract_pass_name)
}

fn extract_pass_name(mut name: &str) -> String {
    if let Some(other) = name.strip_prefix("mlirCreateConversion") {
        name = other.into();
    }

    if let Some(other) = name.strip_suffix("ConversionPass") {
        name = other.into();
    }

    name.into()
}
