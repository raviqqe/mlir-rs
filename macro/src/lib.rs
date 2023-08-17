mod attribute;
mod dialect;
mod operation;
mod parse;
mod pass;
mod r#type;
mod utility;

use dialect::DialectMacroInput;
use parse::{DialectOperationSet, IdentifierList};
use proc_macro::TokenStream;
use quote::quote;
use std::error::Error;
use syn::parse_macro_input;

/// Generates a dialect module from a TableGen file.
///
/// # Examples
///
/// ```rust
/// melior_macro::dialect! {
///     name: "func",
///     tablegen: r#"include "mlir/Dialect/Func/IR/FuncOps.td""#
/// }
/// ```
#[proc_macro]
pub fn dialect(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DialectMacroInput);

    convert_result(dialect::generate_dialect(input))
}

#[proc_macro]
pub fn binary_operations(stream: TokenStream) -> TokenStream {
    let set = parse_macro_input!(stream as DialectOperationSet);

    convert_result(operation::generate_binary(set.dialect(), set.identifiers()))
}

#[proc_macro]
pub fn unary_operations(stream: TokenStream) -> TokenStream {
    let set = parse_macro_input!(stream as DialectOperationSet);

    convert_result(operation::generate_unary(set.dialect(), set.identifiers()))
}

#[proc_macro]
pub fn typed_unary_operations(stream: TokenStream) -> TokenStream {
    let set = parse_macro_input!(stream as DialectOperationSet);

    convert_result(operation::generate_typed_unary(
        set.dialect(),
        set.identifiers(),
    ))
}

#[proc_macro]
pub fn type_check_functions(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(r#type::generate(identifiers.identifiers()))
}

#[proc_macro]
pub fn attribute_check_functions(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(attribute::generate(identifiers.identifiers()))
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
