mod conversion_passes;
mod parse;

use parse::IdentifierList;
use proc_macro::TokenStream;
use quote::quote;
use std::error::Error;
use syn::parse_macro_input;

#[proc_macro]
pub fn conversion_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(conversion_passes::generate(identifiers.identifiers()))
}

fn convert_result(result: Result<TokenStream, Box<dyn Error>>) -> TokenStream {
    result.unwrap_or_else(|error| {
        let message = error.to_string();

        quote! { compile_error!(#message) }.into()
    })
}
