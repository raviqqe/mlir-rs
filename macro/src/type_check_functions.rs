use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;

const FLOAT_8E5M2_PATTERN: &str = "8_e_5_m_2";

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
        "bf_16" | "f_16" | "f_32" | "f_64" => name.replace("_", ""),
        name => name.replace(FLOAT_8E5M2_PATTERN, "8e5m2").into(),
    }
}
