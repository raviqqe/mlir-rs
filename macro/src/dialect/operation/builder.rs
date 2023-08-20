use super::{super::utility::sanitize_snake_case_name, FieldKind, Operation, OperationField};
use convert_case::{Case, Casing};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};

#[derive(Debug)]
struct TypeStateItem {
    pub(crate) field_name: String,
    pub(crate) yes: Ident,
    pub(crate) no: Ident,
    pub(crate) t: Ident,
}

impl TypeStateItem {
    pub fn new(class_name: &str, field_name: &str) -> Self {
        let new_field_name = field_name.to_string().to_case(Case::Pascal);
        Self {
            field_name: field_name.to_string(),
            yes: format_ident!("{}__Yes__{}", class_name, new_field_name),
            no: format_ident!("{}__No__{}", class_name, new_field_name),
            t: format_ident!("{}__Any__{}", class_name, new_field_name),
        }
    }
}

#[derive(Debug)]
struct TypeStateList(Vec<TypeStateItem>);

impl TypeStateList {
    pub fn iter(&self) -> impl Iterator<Item = &TypeStateItem> {
        self.0.iter()
    }

    pub fn iter_all_any(&self) -> impl Iterator<Item = &Ident> {
        self.0.iter().map(|i| &i.t)
    }

    pub fn iter_all_any_without(&self, field_name: String) -> impl Iterator<Item = &Ident> {
        self.0.iter().filter_map(move |i| {
            if i.field_name != field_name {
                Some(&i.t)
            } else {
                None
            }
        })
    }

    pub fn iter_set_yes(&self, field_name: String) -> impl Iterator<Item = &Ident> {
        self.0.iter().map(move |item| {
            if item.field_name == field_name {
                &item.yes
            } else {
                &item.t
            }
        })
    }

    pub fn iter_set_no(&self, field_name: String) -> impl Iterator<Item = &Ident> {
        self.0.iter().map(move |item| {
            if item.field_name == field_name {
                &item.no
            } else {
                &item.t
            }
        })
    }

    pub fn iter_all_yes(&self) -> impl Iterator<Item = &Ident> {
        self.0.iter().map(|i| &i.yes)
    }

    pub fn iter_all_no(&self) -> impl Iterator<Item = &Ident> {
        self.0.iter().map(|i| &i.no)
    }
}

pub struct OperationBuilder<'o, 'c> {
    pub(crate) operation: &'c Operation<'o>,
    type_state: TypeStateList,
}

impl<'o, 'c> OperationBuilder<'o, 'c> {
    pub fn new(operation: &'c Operation<'o>) -> Self {
        let type_state = Self::create_type_state(operation);
        Self {
            operation,
            type_state,
        }
    }

    pub fn methods<'a, 's: 'a>(
        &'s self,
        field_names: &'a [Ident],
        phantoms: &'a [TokenStream],
    ) -> impl Iterator<Item = TokenStream> + 'a {
        let builder_ident = format_ident!("{}Builder", self.operation.class_name);
        self.operation.fields.iter().map(move |field| {
            let name = sanitize_snake_case_name(field.name);
            let st = &field.kind.param_type();
            let args = quote! { #name: #st };
            let add = format_ident!("add_{}s", field.kind.as_str());

            let add_args = {
                let mlir_ident = {
                    let name_str = &field.name;
                    quote! { ::melior::ir::Identifier::new(self.context, #name_str) }
                };

                // Argument types can be singular and variadic, but add functions in melior
                // are always variadic, so we need to create a slice or vec for singular
                // arguments
                match &field.kind {
                    FieldKind::Element { constraint, .. } => {
                        if constraint.is_variable_length() && !constraint.is_optional() {
                            quote! { #name }
                        } else {
                            quote! { &[#name] }
                        }
                    }
                    FieldKind::Attribute { .. } => {
                        quote! { &[(#mlir_ident, #name.into())] }
                    }
                    FieldKind::Successor { constraint, .. } => {
                        if constraint.is_variadic() {
                            quote! { #name }
                        } else {
                            quote! { &[#name] }
                        }
                    }
                    FieldKind::Region { constraint, .. } => {
                        if constraint.is_variadic() {
                            quote! { #name }
                        } else {
                            quote! { vec![#name] }
                        }
                    }
                }
            };

            if field.kind.is_optional() {
                let iter_all_any = self.type_state.iter_all_any().collect::<Vec<_>>();
                quote! {
                    impl<'c, #(#iter_all_any),*> #builder_ident<'c, #(#iter_all_any),*> {
                        pub fn #name(mut self, #args) -> #builder_ident<'c, #(#iter_all_any),*> {
                            self.builder = self.builder.#add(#add_args);
                            self
                        }
                    }
                }
            } else if field.kind.is_result() && self.operation.can_infer_type {
                quote!()
            } else {
                let iter_all_any_without =
                    self.type_state.iter_all_any_without(field.name.to_string());
                let iter_set_yes = self.type_state.iter_set_yes(field.name.to_string());
                let iter_set_no = self.type_state.iter_set_no(field.name.to_string());
                quote! {
                    impl<'c, #(#iter_all_any_without),*> #builder_ident<'c, #(#iter_set_no),*> {
                        pub fn #name(mut self, #args) -> #builder_ident<'c, #(#iter_set_yes),*> {
                            self.builder = self.builder.#add(#add_args);
                            let Self { context, mut builder, #(#field_names),* } = self;
                            #builder_ident {
                                context,
                                builder,
                                #(#phantoms),*
                            }
                        }
                    }
                }
            }
        })
    }

    pub fn builder(&self) -> TokenStream {
        let type_state_structs = self.type_state_structs();
        let builder_ident = format_ident!("{}Builder", self.operation.class_name);

        let field_names = self
            .type_state
            .iter()
            .map(|field| sanitize_snake_case_name(&field.field_name))
            .collect::<Vec<_>>();

        let fields = self
            .type_state
            .iter_all_any()
            .zip(field_names.iter())
            .map(|(g, n)| {
                Some(quote! {
                    #[doc(hidden)]
                    #n: ::std::marker::PhantomData<#g>
                })
            });

        let phantoms: Vec<_> = field_names
            .iter()
            .map(|n| quote! { #n: ::std::marker::PhantomData })
            .collect();

        let methods = self.methods(field_names.as_slice(), phantoms.as_slice());

        let new = {
            let name_str = &self.operation.full_name;
            let iter_all_no = self.type_state.iter_all_no();
            let phantoms = phantoms.clone();
            quote! {
                impl<'c> #builder_ident<'c, #(#iter_all_no),*> {
                    pub fn new(location: ::melior::ir::Location<'c>) -> Self {
                        Self {
                            context: unsafe { location.context().to_ref() },
                            builder: ::melior::ir::operation::OperationBuilder::new(#name_str, location),
                            #(#phantoms),*
                        }
                    }
                }
            }
        };

        let build = {
            let iter_all_yes = self.type_state.iter_all_yes();
            let class_name = format_ident!("{}", &self.operation.class_name);
            let err = format!("should be a valid {}", class_name);
            let maybe_infer = if self.operation.can_infer_type {
                quote! { .enable_result_type_inference() }
            } else {
                quote! {}
            };
            quote! {
                impl<'c> #builder_ident<'c, #(#iter_all_yes),*> {
                    pub fn build(self) -> #class_name<'c> {
                        self.builder #maybe_infer.build().try_into().expect(#err)
                    }
                }
            }
        };

        let doc = format!("Builder for {}", self.operation.summary);
        let iter_all_any = self.type_state.iter_all_any();

        quote! {
            #type_state_structs

            #[doc = #doc]
            pub struct #builder_ident <'c, #(#iter_all_any),* > {
                #[doc(hidden)]
                builder: ::melior::ir::operation::OperationBuilder<'c>,
                #[doc(hidden)]
                context: &'c ::melior::Context,
                #(#fields),*
            }

            #new

            #(#methods)*

            #build
        }
    }

    pub fn create_op_builder_fn(&self) -> TokenStream {
        let builder_ident = format_ident!("{}Builder", self.operation.class_name);
        let iter_all_no = self.type_state.iter_all_no();
        quote! {
            pub fn builder(location: ::melior::ir::Location<'c>) -> #builder_ident<'c, #(#iter_all_no),*> {
                #builder_ident::new(location)
            }
        }
    }

    pub fn default_constructor(&self) -> TokenStream {
        let class_name = format_ident!("{}", &self.operation.class_name);
        let name = sanitize_snake_case_name(self.operation.short_name);
        let mut args = Self::required_fields(self.operation)
            .map(|field| {
                let param_type = &field.kind.param_type();
                let param_name = &field.sanitized_name;
                quote! { #param_name: #param_type }
            })
            .collect::<Vec<_>>();
        let builder_calls = Self::required_fields(self.operation).map(|field| {
            let param_name = &field.sanitized_name;
            quote! { .#param_name(#param_name) }
        });
        args.push(quote! { location: ::melior::ir::Location<'c> });

        let doc = format!("Creates a new {}", self.operation.summary);
        quote! {
            #[allow(clippy::too_many_arguments)]
            #[doc = #doc]
            pub fn #name<'c>(#(#args),*) -> #class_name<'c> {
                #class_name::builder(location)#(#builder_calls)*.build()
            }
        }
    }

    fn required_fields<'a, 'b>(
        operation: &'a Operation<'b>,
    ) -> impl Iterator<Item = &'a OperationField<'b>> {
        operation.fields.iter().filter(|field| {
            !field.kind.is_optional() && (!field.kind.is_result() || !operation.can_infer_type)
        })
    }

    fn create_type_state(operation: &'c Operation<'o>) -> TypeStateList {
        TypeStateList(
            Self::required_fields(operation)
                .map(|field| TypeStateItem::new(operation.class_name, field.name))
                .collect(),
        )
    }

    fn type_state_structs(&self) -> TokenStream {
        self.type_state
            .iter()
            .map(|item| {
                let yes = &item.yes;
                let no = &item.no;
                quote! {
                    #[allow(non_camel_case_types)]
                    #[doc(hidden)]
                    pub struct #yes;
                    #[allow(non_camel_case_types)]
                    #[doc(hidden)]
                    pub struct #no;
                }
            })
            .collect()
    }
}
