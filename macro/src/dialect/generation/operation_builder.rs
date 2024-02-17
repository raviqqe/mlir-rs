use crate::dialect::{
    error::Error, operation::OperationBuilder, utility::sanitize_snake_case_identifier,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub fn generate_operation_builder(builder: &OperationBuilder) -> Result<TokenStream, Error> {
    let state_types = builder.type_state().parameters();
    let field_fns = generate_field_fns(builder);
    let new_fn = generate_new_fn(builder)?;
    let build_fn = generate_build_fn(builder)?;

    let identifier = builder.identifier();
    let doc = format!("A builder for {}", builder.operation().summary()?);
    let type_arguments = builder.type_state().parameters();

    Ok(quote! {
        #[doc = #doc]
        pub struct #identifier<'c, #(#type_arguments),*> {
            builder: ::melior::ir::operation::OperationBuilder<'c>,
            context: &'c ::melior::Context,
            _state: ::std::marker::PhantomData<(#(#state_types),*)>,
        }

        #new_fn

        #(#field_fns)*

        #build_fn
    })
}

// TODO Split this function for different kinds of fields.
fn generate_field_fns(builder: &OperationBuilder) -> Vec<TokenStream> {
    builder.operation().fields().map(move |field| {
        let builder_identifier = builder.identifier();
        let identifier = field.singular_identifier();
        let parameter_type = field.parameter_type();
        let argument = quote! { #identifier: #parameter_type };
        let add = format_ident!("add_{}", field.plural_kind_identifier());

        // Argument types can be singular and variadic. But `add` functions in Melior
        // are always variadic, so we need to create a slice or `Vec` for singular
        // arguments.
        let add_arguments = field.add_arguments(identifier);

        if field.is_optional() {
            let parameters = builder.type_state().parameters().collect::<Vec<_>>();

            quote! {
                impl<'c, #(#parameters),*> #builder_identifier<'c, #(#parameters),*> {
                    pub fn #identifier(mut self, #argument) -> #builder_identifier<'c, #(#parameters),*> {
                        self.builder = self.builder.#add(#add_arguments);
                        self
                    }
                }
            }
        } else if field.is_result() && builder.operation().can_infer_type() {
            quote!()
        } else {
            let parameters = builder.type_state().parameters_without(field.name());
            let arguments_set = builder.type_state().arguments_set(field.name(), true);
            let arguments_unset = builder.type_state().arguments_set(field.name(), false);

            quote! {
                impl<'c, #(#parameters),*> #builder_identifier<'c, #(#arguments_unset),*> {
                    pub fn #identifier(self, #argument) -> #builder_identifier<'c, #(#arguments_set),*> {
                        #builder_identifier {
                            context: self.context,
                            builder: self.builder.#add(#add_arguments),
                            _state: Default::default(),
                        }
                    }
                }
            }
        }
    }).collect()
}

fn generate_build_fn(builder: &OperationBuilder) -> Result<TokenStream, Error> {
    let builder_identifier = builder.identifier();
    let arguments = builder.type_state().arguments_all_set(true);
    let operation_identifier = format_ident!("{}", &builder.operation().name());
    let error = format!("should be a valid {operation_identifier}");
    let maybe_infer = builder
        .operation()
        .can_infer_type()
        .then_some(quote! { .enable_result_type_inference() });

    Ok(quote! {
        impl<'c> #builder_identifier<'c, #(#arguments),*> {
            pub fn build(self) -> #operation_identifier<'c> {
                self.builder #maybe_infer.build().expect("valid operation").try_into().expect(#error)
            }
        }
    })
}

fn generate_new_fn(builder: &OperationBuilder) -> Result<TokenStream, Error> {
    let builder_ident = builder.identifier();
    let name = &builder.operation().full_operation_name()?;
    let arguments = builder.type_state().arguments_all_set(false);

    Ok(quote! {
        impl<'c> #builder_ident<'c, #(#arguments),*> {
            pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
                Self {
                    context,
                    builder: ::melior::ir::operation::OperationBuilder::new(#name, location),
                    _state: Default::default(),
                }
            }
        }
    })
}

pub fn generate_operation_builder_fn(builder: &OperationBuilder) -> TokenStream {
    let builder_ident = builder.identifier();
    let arguments = builder.type_state().arguments_all_set(false);

    quote! {
        /// Creates a builder.
        pub fn builder(
            context: &'c ::melior::Context,
            location: ::melior::ir::Location<'c>
        ) -> #builder_ident<'c, #(#arguments),*> {
            #builder_ident::new(context, location)
        }
    }
}

pub fn generate_default_constructor(builder: &OperationBuilder) -> Result<TokenStream, Error> {
    let identifier = format_ident!("{}", &builder.operation().name());
    let name = sanitize_snake_case_identifier(builder.operation().operation_name()?)?;
    let arguments = builder
        .operation()
        .required_fields()
        .map(|field| {
            let parameter_type = &field.parameter_type();
            let parameter_name = &field.singular_identifier();

            quote! { #parameter_name: #parameter_type }
        })
        .chain([quote! { location: ::melior::ir::Location<'c> }])
        .collect::<Vec<_>>();
    let builder_calls = builder
        .operation()
        .required_fields()
        .map(|field| {
            let parameter_name = &field.singular_identifier();

            quote! { .#parameter_name(#parameter_name) }
        })
        .collect::<Vec<_>>();

    let doc = format!("Creates a {}", builder.operation().summary()?);

    Ok(quote! {
        #[allow(clippy::too_many_arguments)]
        #[doc = #doc]
        pub fn #name<'c>(context: &'c ::melior::Context, #(#arguments),*) -> #identifier<'c> {
            #identifier::builder(context, location)#(#builder_calls)*.build()
        }
    })
}
