mod parse;
mod passes;
mod type_check_functions;

use parse::IdentifierList;
use proc_macro::TokenStream;
use quote::quote;
use std::error::Error;
use syn::parse_macro_input;

#[proc_macro]
pub fn type_check_functions(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(type_check_functions::generate(identifiers.identifiers()))
}

#[proc_macro]
pub fn conversion_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(passes::generate(identifiers.identifiers(), |mut name| {
        if let Some(other) = name.strip_prefix("Conversion") {
            name = other.into();
        }

        if let Some(other) = name.strip_prefix("Convert") {
            name = other.into();
        }

        if let Some(other) = name.strip_suffix("ConversionPass") {
            name = other.into();
        }

        name.into()
    }))
}

#[proc_macro]
pub fn transform_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(passes::generate(identifiers.identifiers(), |name| {
        name.strip_prefix("Transforms").unwrap_or(name).into()
    }))
}

fn convert_result(result: Result<TokenStream, Box<dyn Error>>) -> TokenStream {
    result.unwrap_or_else(|error| {
        let message = error.to_string();

        quote! { compile_error!(#message) }.into()
    })
}
