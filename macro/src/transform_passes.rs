use crate::passes;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use std::error::Error;

pub fn generate(identifiers: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    passes::generate(identifiers, extract_pass_name)
}

fn extract_pass_name(name: &str) -> String {
    name.strip_prefix("mlirCreateTransforms")
        .unwrap_or(name)
        .into()
}
