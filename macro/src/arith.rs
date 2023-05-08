use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;

pub fn generate_binary_operators(names: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for name in names {
        let document = format!(" Creates an operation of `arith.{}`.", name);
        let operation_name = format!("arith.{}", name);

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #name<'c>(
                lhs: crate::ir::Value,
                rhs: crate::ir::Value,
                location: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                binary_operator(#operation_name, lhs, rhs, location)
            }
        }));
    }

    stream.extend(TokenStream::from(quote! {
        fn binary_operator<'c>(
            name: &str,
            lhs: crate::ir::Value,
            rhs: crate::ir::Value,
            location: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::Builder::new(name, location)
                .add_operands(&[lhs, rhs])
                .enable_result_type_inference()
                .build()
        }
    }));

    Ok(stream)
}
