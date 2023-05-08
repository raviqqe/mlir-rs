use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;

pub fn generate(identifiers: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for identifier in identifiers {
        let name = identifier
            .to_string()
            .strip_prefix("mlirTypeIsA")
            .unwrap()
            .to_case(Case::Snake);

        let function_name = Ident::new(&format!("is_{}", &name), identifier.span());
        let document = format!(" Returns `true` if a type is `{}`.", name);

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #function_name(&self) -> bool {
                unsafe { #name(self.to_raw()) }
            }
        }));
    }

    Ok(stream)
}
