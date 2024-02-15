mod type_state_item;
mod type_state_list;

use self::{type_state_item::TypeStateItem, type_state_list::TypeStateList};
use super::{
    super::{error::Error, utility::sanitize_snake_case_name},
    operation_field::OperationFieldLike,
    Operation,
};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};

pub struct OperationBuilder<'o> {
    operation: &'o Operation<'o>,
    identifier: Ident,
    type_state: TypeStateList,
}

impl<'o> OperationBuilder<'o> {
    pub fn new(operation: &'o Operation<'o>) -> Result<Self, Error> {
        Ok(Self {
            operation,
            identifier: format_ident!("{}Builder", operation.class_name()?),
            type_state: Self::create_type_state(operation),
        })
    }

    pub fn operation(&self) -> &Operation {
        self.operation
    }

    pub fn identifier(&self) -> &Ident {
        &self.identifier
    }

    pub fn type_state(&self) -> &TypeStateList {
        &self.type_state
    }

    pub fn create_builder_fns<'a>(
        &'a self,
        field_names: &'a [Ident],
        phantoms: &'a [TokenStream],
    ) -> impl Iterator<Item = Result<TokenStream, Error>> + 'a {
        self.operation.fields().map(move |field| {
            let builder_ident = self.identifier();
            let name = sanitize_snake_case_name(field.name())?;
            let parameter_type = field.parameter_type();
            let argument = quote! { #name: #parameter_type };
            let add = format_ident!("add_{}", field.plural_identifier());

            // Argument types can be singular and variadic, but add functions in melior
            // are always variadic, so we need to create a slice or vec for singular
            // arguments
            let add_arguments = field.add_arguments(&name);

            Ok(if field.is_optional() {
                let parameters = self.type_state.parameters().collect::<Vec<_>>();

                quote! {
                    impl<'c, #(#parameters),*> #builder_ident<'c, #(#parameters),*> {
                        pub fn #name(mut self, #argument) -> #builder_ident<'c, #(#parameters),*> {
                            self.builder = self.builder.#add(#add_arguments);
                            self
                        }
                    }
                }
            } else if field.is_result() && self.operation.can_infer_type {
                quote!()
            } else {
                let parameters = self.type_state.parameters_without(field.name());
                let arguments_set = self.type_state.arguments_set(field.name(), true);
                let arguments_unset = self.type_state.arguments_set(field.name(), false);

                quote! {
                    impl<'c, #(#parameters),*> #builder_ident<'c, #(#arguments_unset),*> {
                        pub fn #name(mut self, #argument) -> #builder_ident<'c, #(#arguments_set),*> {
                            self.builder = self.builder.#add(#add_arguments);
                            let Self { context, mut builder, #(#field_names),* } = self;
                            #builder_ident {
                                context,
                                builder,
                                #(#phantoms),*
                            }
                        }
                    }
                }
            })
        })
    }

    pub fn create_build_fn(&self) -> Result<TokenStream, Error> {
        let builder_ident = self.builder_identifier()?;
        let arguments = self.type_state.arguments_all_set(true);
        let class_name = format_ident!("{}", &self.operation.class_name()?);
        let error = format!("should be a valid {class_name}");
        let maybe_infer = self
            .operation
            .can_infer_type
            .then_some(quote! { .enable_result_type_inference() });

        Ok(quote! {
            impl<'c> #builder_ident<'c, #(#arguments),*> {
                pub fn build(self) -> #class_name<'c> {
                    self.builder #maybe_infer.build().expect("valid operation").try_into().expect(#error)
                }
            }
        })
    }

    pub fn create_new_fn(&self, phantoms: &[TokenStream]) -> Result<TokenStream, Error> {
        let builder_ident = self.builder_identifier()?;
        let name = &self.operation.full_name()?;
        let arguments = self.type_state.arguments_all_set(false);

        Ok(quote! {
            impl<'c> #builder_ident<'c, #(#arguments),*> {
                pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
                    Self {
                        context,
                        builder: ::melior::ir::operation::OperationBuilder::new( #name, location),
                        #(#phantoms),*
                    }
                }
            }
        })
    }

    pub fn create_op_builder_fn(&self) -> Result<TokenStream, Error> {
        let builder_ident = self.builder_identifier()?;
        let arguments = self.type_state.arguments_all_set(false);

        Ok(quote! {
            pub fn builder(
                context: &'c ::melior::Context,
                location: ::melior::ir::Location<'c>
            ) -> #builder_ident<'c, #(#arguments),*> {
                #builder_ident::new(context, location)
            }
        })
    }

    pub fn create_default_constructor(&self) -> Result<TokenStream, Error> {
        let class_name = format_ident!("{}", &self.operation.class_name()?);
        let name = sanitize_snake_case_name(self.operation.short_name()?)?;
        let arguments = Self::required_fields(self.operation)
            .map(|field| {
                let parameter_type = &field.parameter_type();
                let parameter_name = &field.sanitized_name();

                quote! { #parameter_name: #parameter_type }
            })
            .chain([quote! { location: ::melior::ir::Location<'c> }])
            .collect::<Vec<_>>();
        let builder_calls = Self::required_fields(self.operation)
            .map(|field| {
                let parameter_name = &field.sanitized_name();

                quote! { .#parameter_name(#parameter_name) }
            })
            .collect::<Vec<_>>();

        let doc = format!("Creates a new {}", self.operation.summary()?);

        Ok(quote! {
            #[allow(clippy::too_many_arguments)]
            #[doc = #doc]
            pub fn #name<'c>(context: &'c ::melior::Context, #(#arguments),*) -> #class_name<'c> {
                #class_name::builder(context, location)#(#builder_calls)*.build()
            }
        })
    }

    fn required_fields<'a>(
        operation: &'a Operation,
    ) -> impl Iterator<Item = &'a dyn OperationFieldLike> {
        operation.fields().filter(|field| {
            (!field.is_result() || !operation.can_infer_type) && !field.is_optional()
        })
    }

    fn create_type_state(operation: &'o Operation<'o>) -> TypeStateList {
        TypeStateList::new(
            Self::required_fields(operation)
                .enumerate()
                .map(|(index, field)| TypeStateItem::new(index, field.name().to_string()))
                .collect(),
        )
    }
}
