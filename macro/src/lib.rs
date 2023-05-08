mod attribute_list;
mod instruction;
mod utility;

use self::attribute_list::AttributeList;
use proc_macro::TokenStream;
use quote::quote;
use std::error::Error;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn instruction(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let attributes = parse_macro_input!(attributes as AttributeList);
    let function = parse_macro_input!(item as ItemFn);

    convert_result(instruction::generate(&attributes, &function))
}

fn convert_result(result: Result<TokenStream, Box<dyn Error>>) -> TokenStream {
    result.unwrap_or_else(|error| {
        let message = error.to_string();

        quote! { compile_error!(#message) }.into()
    })
}
