use super::OperationField;
use crate::dialect::{
    error::Error,
    types::SuccessorConstraint,
    utility::{generate_iterator_type, generate_result_type, sanitize_snake_case_identifier},
};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{parse_quote, Ident, Type};

#[derive(Debug)]
pub struct Successor<'a> {
    name: &'a str,
    singular_identifier: Ident,
    constraint: SuccessorConstraint<'a>,
}

impl<'a> Successor<'a> {
    pub fn new(name: &'a str, constraint: SuccessorConstraint<'a>) -> Result<Self, Error> {
        Ok(Self {
            name,
            singular_identifier: sanitize_snake_case_identifier(name)?,
            constraint,
        })
    }

    pub fn is_variadic(&self) -> bool {
        self.constraint.is_variadic()
    }
}

impl OperationField for Successor<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn singular_identifier(&self) -> &Ident {
        &self.singular_identifier
    }

    fn plural_kind_identifier(&self) -> Ident {
        Ident::new("successors", Span::call_site())
    }

    fn parameter_type(&self) -> Type {
        let r#type: Type = parse_quote!(&::melior::ir::Block<'c>);

        if self.is_variadic() {
            parse_quote!(&[#r#type])
        } else {
            r#type
        }
    }

    fn return_type(&self) -> Type {
        let r#type: Type = parse_quote!(::melior::ir::BlockRef<'c, '_>);

        if self.is_variadic() {
            generate_iterator_type(r#type)
        } else {
            generate_result_type(r#type)
        }
    }

    fn is_optional(&self) -> bool {
        false
    }

    fn is_result(&self) -> bool {
        false
    }

    fn add_arguments(&self, name: &Ident) -> TokenStream {
        if self.is_variadic() {
            quote! { #name }
        } else {
            quote! { &[#name] }
        }
    }
}