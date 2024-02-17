use super::{OperationField, OperationFieldLike};
use crate::dialect::{
    error::Error,
    types::{RegionConstraint, TypeConstraint},
    utility::sanitize_snake_case_identifier,
};
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, Type};

#[derive(Debug)]
pub struct OperationResult<'a> {
    name: &'a str,
    singular_identifier: Ident,
    constraint: RegionConstraint<'a>,
}

impl<'a> OperationResult<'a> {
    fn new(name: &'a str, constraint: TypeConstraint<'a>) -> Result<Self, Error> {
        Ok(Self {
            name,
            singular_identifier: sanitize_snake_case_identifier(name)?,
            constraint,
        })
    }
}

impl OperationFieldLike for OperationResult<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn singular_identifier(&self) -> &Ident {
        &self.singular_identifier
    }

    fn plural_kind_identifier(&self) -> Ident {
        Ident::new("results", Span::call_site())
    }

    fn parameter_type(&self) -> Type {
        self.kind.parameter_type()
    }

    fn return_type(&self) -> Type {
        self.kind.return_type()
    }

    fn is_optional(&self) -> bool {
        self.kind.is_optional()
    }

    fn is_result(&self) -> bool {
        true
    }

    fn add_arguments(&self, name: &Ident) -> TokenStream {
        if self.constraint.is_unfixed() && !self.constraint.is_optional() {
            quote! { #name }
        } else {
            quote! { &[#name] }
        }
    }
}
