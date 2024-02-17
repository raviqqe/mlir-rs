use super::{element_kind::ElementKind, field_kind::FieldKind, SequenceInfo, VariadicKind};
use crate::dialect::{
    error::Error, types::TypeConstraint, utility::sanitize_snake_case_identifier,
};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use syn::Type;

// TODO Rename this `OperationField`.
pub trait OperationFieldLike {
    fn name(&self) -> &str;
    fn singular_identifier(&self) -> &Ident;
    fn plural_kind_identifier(&self) -> Ident;
    fn parameter_type(&self) -> Type;
    fn return_type(&self) -> Type;
    fn is_optional(&self) -> bool;
    // TODO Remove this.
    fn is_result(&self) -> bool;
    fn add_arguments(&self, name: &Ident) -> TokenStream;
}

#[derive(Debug)]
pub struct OperationField<'a> {
    pub(crate) name: &'a str,
    pub(crate) plural_identifier: Ident,
    pub(crate) sanitized_name: Ident,
    pub(crate) kind: FieldKind<'a>,
}

impl OperationFieldLike for OperationField<'_> {
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
        matches!(
            self.kind,
            FieldKind::Element {
                kind: ElementKind::Result,
                ..
            }
        )
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

impl<'a> OperationField<'a> {
    fn new(name: &'a str, kind: FieldKind<'a>) -> Result<Self, Error> {
        Ok(Self {
            name,
            plural_identifier: match kind {
                FieldKind::Element { kind, .. } => format_ident!("{}s", kind.as_str()),
            },
            sanitized_name: sanitize_snake_case_identifier(name)?,
            kind,
        })
    }

    pub fn new_element(
        name: &'a str,
        constraint: TypeConstraint<'a>,
        kind: ElementKind,
        sequence_info: SequenceInfo,
        variadic_kind: VariadicKind,
    ) -> Result<Self, Error> {
        Self::new(
            name,
            FieldKind::Element {
                kind,
                constraint,
                sequence_info,
                variadic_kind,
            },
        )
    }
}
