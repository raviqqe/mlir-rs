use crate::passes;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use std::error::Error;

pub fn generate(identifiers: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    passes::generate(identifiers, |name| {
        name.strip_prefix("Transforms").unwrap_or(name).into()
    })
}
