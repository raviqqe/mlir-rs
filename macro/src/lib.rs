mod arith;
mod parse;
mod pass;
mod type_check_functions;

use parse::IdentifierList;
use proc_macro::TokenStream;
use quote::quote;
use std::error::Error;
use syn::parse_macro_input;

#[proc_macro]
pub fn arith_binary_operators(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(arith::generate_binary_operators(identifiers.identifiers()))
}

#[proc_macro]
pub fn type_check_functions(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(type_check_functions::generate(identifiers.identifiers()))
}

#[proc_macro]
pub fn async_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(pass::generate(identifiers.identifiers(), |name| {
        name.strip_prefix("Async").unwrap().into()
    }))
}

#[proc_macro]
pub fn conversion_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(pass::generate(identifiers.identifiers(), |mut name| {
        name = name.strip_prefix("Conversion").unwrap();
        name = name.strip_prefix("Convert").unwrap_or(name);
        name.strip_suffix("ConversionPass").unwrap_or(name).into()
    }))
}

#[proc_macro]
pub fn gpu_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(pass::generate(identifiers.identifiers(), |name| {
        name.strip_prefix("GPU").unwrap().into()
    }))
}

#[proc_macro]
pub fn transform_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(pass::generate(identifiers.identifiers(), |name| {
        name.strip_prefix("Transforms").unwrap().into()
    }))
}

#[proc_macro]
pub fn linalg_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(pass::generate(identifiers.identifiers(), |name| {
        name.strip_prefix("Linalg").unwrap().into()
    }))
}

#[proc_macro]
pub fn sparse_tensor_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(pass::generate(identifiers.identifiers(), |name| {
        name.strip_prefix("SparseTensor").unwrap().into()
    }))
}

fn convert_result(result: Result<TokenStream, Box<dyn Error>>) -> TokenStream {
    result.unwrap_or_else(|error| {
        let message = error.to_string();

        quote! { compile_error!(#message) }.into()
    })
}
