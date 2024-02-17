use super::{field_kind::FieldKind, OperationElement, OperationField, SequenceInfo, VariadicKind};
use crate::dialect::{
    error::Error, types::TypeConstraint, utility::sanitize_snake_case_identifier,
};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use syn::Type;

#[derive(Debug)]
pub struct Operand<'a> {
    pub(crate) name: &'a str,
    pub(crate) plural_identifier: Ident,
    pub(crate) sanitized_name: Ident,
    pub(crate) kind: FieldKind<'a>,
}

impl<'a> Operand<'a> {
    pub fn new(
        name: &'a str,
        constraint: TypeConstraint<'a>,
        sequence_info: SequenceInfo,
        variadic_kind: VariadicKind,
    ) -> Result<Self, Error> {
        Ok(Self {
            name,
            plural_identifier: format_ident!("operands"),
            sanitized_name: sanitize_snake_case_identifier(name)?,
            kind: FieldKind::Element {
                constraint,
                sequence_info,
                variadic_kind,
            },
        })
    }
}

impl OperationField for Operand<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn singular_identifier(&self) -> &Ident {
        &self.sanitized_name
    }

    fn plural_kind_identifier(&self) -> Ident {
        self.plural_identifier.clone()
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
        false
    }

    fn add_arguments(&self, name: &Ident) -> TokenStream {
        match &self.kind {
            FieldKind::Element { constraint, .. } => {
                if constraint.is_variadic() {
                    quote! { #name }
                } else {
                    quote! { &[#name] }
                }
            }
        }
    }
}

impl OperationElement for Operand<'_> {
    fn is_variadic(&self) -> bool {
        match &self.kind {
            FieldKind::Element { constraint, .. } => constraint.is_variadic(),
        }
    }

    fn variadic_kind(&self) -> &VariadicKind {
        match &self.kind {
            FieldKind::Element { variadic_kind, .. } => variadic_kind,
        }
    }
}
