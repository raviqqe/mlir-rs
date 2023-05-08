use crate::{attribute_list::AttributeList, utility::parse_crate_path};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;
use syn::{Expr, ExprLit, FnArg, ItemFn, Lit, LitStr};

const RAW_STRING_PREFIX: &str = "r#";

pub fn generate(
    attributes: &AttributeList,
    function: &ItemFn,
) -> Result<TokenStream, Box<dyn Error>> {
    if function
        .sig
        .inputs
        .iter()
        .any(|input| matches!(input, FnArg::Receiver(_)))
    {
        return Err("receiver not supported".into());
    } else if function.sig.abi.is_some() {
        return Err("custom function ABI not supported".into());
    } else if !function.sig.generics.params.is_empty() {
        return Err("generic function not supported".into());
    } else if function.sig.asyncness.is_some() {
        return Err("async function not supported".into());
    }

    let crate_path = parse_crate_path(attributes)?;
    let ident = &function.sig.ident;
    let ident_string = ident
        .to_string()
        .strip_prefix(RAW_STRING_PREFIX)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| ident.to_string());
    let visibility = &function.vis;
    let variable_name = Ident::new(
        &(ident_string.clone() + "_instruction"),
        function.sig.ident.span(),
    );
    let test_module_name = Ident::new(&(ident_string + "_tests"), function.sig.ident.span());
    let name_string = Expr::Lit(ExprLit {
        attrs: Vec::new(),
        lit: Lit::Str(LitStr::new(&ident.to_string(), ident.span())),
    });

    Ok(quote! {
        #visibility fn #variable_name() -> #crate_path::Instruction {
            let stream = quote::quote!(#function);
            let function = syn::parse2::<syn::ItemFn>(stream).unwrap();

            #crate_path::Instruction::new(#name_string, function)
        }

        #[cfg(test)]
        mod #test_module_name {
            use super::*;

            #[test]
            fn no_panic() {
                #variable_name();
            }
        }

        #function
    }
    .into())
}
