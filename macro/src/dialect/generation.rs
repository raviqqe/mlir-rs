mod attribute_accessor;
mod field_accessor;
mod operation_builder;

use self::{
    attribute_accessor::generate_attribute_accessors, field_accessor::generate_accessor,
    operation_builder::generate_operation_builder,
};
use super::operation::{Operation, OperationBuilder};
use crate::dialect::error::Error;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub fn generate_operation(operation: &Operation) -> Result<TokenStream, Error> {
    let summary = operation.summary()?;
    let description = operation.description()?;
    let class_name = format_ident!("{}", operation.class_name()?);
    let name = &operation.full_name()?;

    let field_accessors = operation
        .operation_fields()
        .map(generate_accessor)
        .collect::<Result<Vec<_>, _>>()?;
    let attribute_accessors = operation
        .attributes()
        .map(generate_attribute_accessors)
        .collect::<Result<Vec<_>, _>>()?;

    let builder = OperationBuilder::new(operation);
    let builder_tokens = generate_operation_builder(&builder)?;
    let builder_fn = builder.create_op_builder_fn()?;
    let default_constructor = builder.create_default_constructor()?;

    Ok(quote! {
        #[doc = #summary]
        #[doc = "\n\n"]
        #[doc = #description]
        pub struct #class_name<'c> {
            operation: ::melior::ir::operation::Operation<'c>,
        }

        impl<'c> #class_name<'c> {
            pub fn name() -> &'static str {
                #name
            }

            pub fn operation(&self) -> &::melior::ir::operation::Operation<'c> {
                &self.operation
            }

            #builder_fn

            #(#field_accessors)*
            #(#attribute_accessors)*
        }

        #builder_tokens

        #default_constructor

        impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for #class_name<'c> {
            type Error = ::melior::Error;

            fn try_from(
                operation: ::melior::ir::operation::Operation<'c>,
            ) -> Result<Self, Self::Error> {
                // TODO Check an operation name.
                Ok(Self { operation })
            }
        }

        impl<'c> From<#class_name<'c>> for ::melior::ir::operation::Operation<'c> {
            fn from(operation: #class_name<'c>) -> ::melior::ir::operation::Operation<'c> {
                operation.operation
            }
        }
    })
}
