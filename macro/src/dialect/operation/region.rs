use super::{OperationFieldLike, SequenceInfo};
use crate::dialect::{
    error::Error,
    types::RegionConstraint,
    utility::{generate_iterator_type, generate_result_type, sanitize_snake_case_identifier},
};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{parse_quote, Ident, Type};

#[derive(Debug)]
pub struct Region<'a> {
    name: &'a str,
    singular_identifier: Ident,
    constraint: RegionConstraint<'a>,
    sequence_info: SequenceInfo,
}

impl<'a> Region<'a> {
    pub fn new(
        name: &'a str,
        constraint: RegionConstraint<'a>,
        sequence_info: SequenceInfo,
    ) -> Result<Self, Error> {
        Ok(Self {
            name,
            singular_identifier: sanitize_snake_case_identifier(name)?,
            constraint,
            sequence_info,
        })
    }

    pub fn is_variadic(&self) -> bool {
        self.constraint.is_variadic()
    }

    pub fn sequence_info(&self) -> &SequenceInfo {
        &self.sequence_info
    }
}

impl OperationFieldLike for Region<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn singular_identifier(&self) -> &Ident {
        &self.singular_identifier
    }

    fn plural_kind_identifier(&self) -> Ident {
        Ident::new("regions", Span::call_site())
    }

    fn parameter_type(&self) -> Type {
        let r#type: Type = parse_quote!(::melior::ir::Region<'c>);

        if self.is_variadic() {
            parse_quote!(Vec<#r#type>)
        } else {
            r#type
        }
    }

    fn return_type(&self) -> Type {
        let r#type: Type = parse_quote!(::melior::ir::RegionRef<'c, '_>);

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
            quote! { vec![#name] }
        }
    }
}
