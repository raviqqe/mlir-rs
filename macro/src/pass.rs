use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;

const CREATE_FUNCTION_PREFIX: &str = "mlirCreate";

pub fn generate(
    identifiers: &[Ident],
    extract_pass_name: impl Fn(&str) -> String,
) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for identifier in identifiers {
        let name = extract_pass_name(
            identifier
                .to_string()
                .strip_prefix(CREATE_FUNCTION_PREFIX)
                .unwrap(),
        );

        let function_name = Ident::new(&name.to_case(Case::Snake), identifier.span());
        let document = format!(" Creates a `{}` pass.", name);

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #function_name() -> crate::pass::Pass {
                crate::pass::Pass::__private_from_raw_fn(mlir_sys::#identifier)
            }
        }));
    }

    for identifier in identifiers {
        let name = identifier.to_string();
        let name = name.strip_prefix(CREATE_FUNCTION_PREFIX).unwrap();

        let foreign_function_name =
            Ident::new(&("mlirRegister".to_owned() + name), identifier.span());
        let name = extract_pass_name(name);
        let function_name = Ident::new(
            &("register_".to_owned() + &name.to_case(Case::Snake)),
            identifier.span(),
        );
        let document = format!(" Registers a `{}` pass.", name);

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #function_name() {
                unsafe { mlir_sys::#foreign_function_name() }
            }
        }));
    }

    Ok(stream)
}
