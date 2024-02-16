use crate::dialect::{
    error::Error, operation::OperationBuilder, utility::sanitize_snake_case_name,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

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

pub fn generate_default_constructor(builder: &OperationBuilder) -> Result<TokenStream, Error> {
    let class_name = format_ident!("{}", &builder.operation().class_name()?);
    let name = sanitize_snake_case_name(builder.operation().short_name()?)?;
    let arguments = builder
        .operation()
        .required_fields()
        .map(|field| {
            let parameter_type = &field.parameter_type();
            let parameter_name = &field.sanitized_name();

            quote! { #parameter_name: #parameter_type }
        })
        .chain([quote! { location: ::melior::ir::Location<'c> }])
        .collect::<Vec<_>>();
    let builder_calls = builder
        .operation()
        .required_fields()
        .map(|field| {
            let parameter_name = &field.sanitized_name();

            quote! { .#parameter_name(#parameter_name) }
        })
        .collect::<Vec<_>>();

    let doc = format!("Creates a new {}", builder.operation().summary()?);

    Ok(quote! {
        #[allow(clippy::too_many_arguments)]
        #[doc = #doc]
        pub fn #name<'c>(context: &'c ::melior::Context, #(#arguments),*) -> #class_name<'c> {
            #class_name::builder(context, location)#(#builder_calls)*.build()
        }
    })
}
