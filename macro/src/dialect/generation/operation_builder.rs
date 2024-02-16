use crate::dialect::{
    error::Error, operation::OperationBuilder, utility::sanitize_snake_case_name,
};
use proc_macro2::TokenStream;
use quote::quote;

pub fn generate_operation_builder(builder: &OperationBuilder) -> Result<TokenStream, Error> {
    let field_names = builder
        .type_state()
        .field_names()
        .map(sanitize_snake_case_name)
        .collect::<Result<Vec<_>, _>>()?;

    let phantom_fields =
        builder
            .type_state()
            .parameters()
            .zip(&field_names)
            .map(|(r#type, name)| {
                quote! {
                    #name: ::std::marker::PhantomData<#r#type>
                }
            });

    let phantom_arguments = field_names
        .iter()
        .map(|name| quote! { #name: ::std::marker::PhantomData })
        .collect::<Vec<_>>();

    let builder_fns = builder
        .create_builder_fns(&field_names, phantom_arguments.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let new_fn = builder.create_new_fn(phantom_arguments.as_slice())?;
    let build_fn = builder.create_build_fn()?;

    let builder_identifier = builder.identifier();
    let doc = format!("A builder for {}", builder.operation().summary()?);
    let type_arguments = builder.type_state().parameters();

    Ok(quote! {
        #[doc = #doc]
        pub struct #builder_identifier<'c, #(#type_arguments),*> {
            builder: ::melior::ir::operation::OperationBuilder<'c>,
            context: &'c ::melior::Context,
            #(#phantom_fields),*
        }

        #new_fn

        #(#builder_fns)*

        #build_fn
    })
}
