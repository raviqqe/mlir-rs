use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;

pub fn generate_binary(dialect: &Ident, names: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for name in names {
        let document = create_document(dialect, name);
        let operation_name = format!("{}.{}", dialect, name);

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

pub fn generate_unary(dialect: &Ident, names: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for name in names {
        let document = create_document(dialect, name);
        let operation_name = format!("{}.{}", dialect, name);

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #name<'c>(
                value: crate::ir::Value,
                location: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                unary_operator(#operation_name, value, location)
            }
        }));
    }

    stream.extend(TokenStream::from(quote! {
        fn unary_operator<'c>(
            name: &str,
            value: crate::ir::Value,
            location: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::Builder::new(name, location)
                .add_operands(&[value])
                .enable_result_type_inference()
                .build()
        }
    }));

    Ok(stream)
}

pub fn generate_typed_unary(
    dialect: &Ident,
    names: &[Ident],
) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for name in names {
        let document = create_document(dialect, name);
        let operation_name = format!("{}.{}", dialect, name);

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #name<'c>(
                value: crate::ir::Value,
                r#type: crate::ir::Type<'c>,
                location: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                unary_operator(#operation_name, value, r#type, location)
            }
        }));
    }

    stream.extend(TokenStream::from(quote! {
        fn unary_operator<'c>(
            name: &str,
            value: crate::ir::Value,
            r#type: crate::ir::Type<'c>,
            location: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::Builder::new(name, location)
                .add_operands(&[value])
                .add_results(&[r#type])
                .build()
        }
    }));

    Ok(stream)
}

fn create_document(dialect: &Ident, name: &Ident) -> String {
    format!(" Creates an `{}.{}` operation.", dialect, name)
}
