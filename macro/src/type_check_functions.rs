use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use regex::{Captures, Regex};
use std::error::Error;

static FLOAT_8_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"8_e_[0-9]_m_[0-9](_fn)?"#).unwrap());

pub fn generate(identifiers: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for identifier in identifiers {
        let name = map_type_name(
            &identifier
                .to_string()
                .strip_prefix("mlirTypeIsA")
                .unwrap()
                .to_case(Case::Snake),
        );

        let function_name = Ident::new(&format!("is_{}", &name), identifier.span());
        let document = format!(" Returns `true` if a type is `{}`.", name);

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            fn #function_name(&self) -> bool {
                unsafe { mlir_sys::#identifier(self.to_raw()) }
            }
        }));
    }

    Ok(stream)
}

fn map_type_name(name: &str) -> String {
    match name {
        "bf_16" | "f_16" | "f_32" | "f_64" => name.replace('_', ""),
        name => FLOAT_8_PATTERN
            .replace(name, |captures: &Captures| {
                captures.get(0).unwrap().as_str().replace('_', "")
            })
            .to_owned()
            .to_string(),
    }
}
