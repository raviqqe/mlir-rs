use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;

pub fn generate_binary(dialect: &Ident, names: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for name in names {
        let document = create_document(dialect, name);
        let operation_name = create_operation_name(dialect, name);

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #name<'c>(
                context: &'c Context,
                lhs: crate::ir::Value<'c, '_>,
                rhs: crate::ir::Value<'c, '_>,
                location: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                binary_operator(context, #operation_name, lhs, rhs, location)
            }
        }));
    }

    stream.extend(TokenStream::from(quote! {
        fn binary_operator<'c>(
            context: &'c Context,
            name: &str,
            lhs: crate::ir::Value<'c, '_>,
            rhs: crate::ir::Value<'c, '_>,
            location: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::OperationBuilder::new(&context, context, name, location)
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
        let operation_name = create_operation_name(dialect, name);

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #name<'c>(
                context: &'c Context,
                value: crate::ir::Value<'c, '_>,
                location: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                unary_operator(context, #operation_name, value, location)
            }
        }));
    }

    stream.extend(TokenStream::from(quote! {
        fn unary_operator<'c>(
            context: &'c Context,
            name: &str,
            value: crate::ir::Value<'c, '_>,
            location: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::OperationBuilder::new(&context, context, name, location)
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
        let operation_name = create_operation_name(dialect, name);

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #name<'c>(
                context: &'c Context,
                value: crate::ir::Value<'c, '_>,
                r#type: crate::ir::Type<'c>,
                location: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                typed_unary_operator(context, #operation_name, value, r#type, location)
            }
        }));
    }

    stream.extend(TokenStream::from(quote! {
        fn typed_unary_operator<'c>(
            context: &'c Context,
            name: &str,
            value: crate::ir::Value<'c, '_>,
            r#type: crate::ir::Type<'c>,
            location: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::OperationBuilder::new(&context, context, name, location)
                .add_operands(&[value])
                .add_results(&[r#type])
                .build()
        }
    }));

    Ok(stream)
}

fn create_document(dialect: &Ident, name: &Ident) -> String {
    format!(
        " Creates an `{}` operation.",
        create_operation_name(dialect, name)
    )
}

fn create_operation_name(dialect: &Ident, name: &Ident) -> String {
    format!("{}.{}", dialect, name)
}
