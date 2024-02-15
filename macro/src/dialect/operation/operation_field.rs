use super::{element_kind::ElementKind, field_kind::FieldKind, SequenceInfo, VariadicKind};
use crate::dialect::{
    error::Error,
    types::{RegionConstraint, SuccessorConstraint, TypeConstraint},
    utility::sanitize_snake_case_name,
};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::Type;

pub trait OperationFieldV2 {
    fn name(&self) -> &str;
    fn plural_identifier(&self) -> &str;
    fn sanitized_name(&self) -> &Ident;
    fn parameter_type(&self) -> Type;
    fn return_type(&self) -> Type;
    fn is_optional(&self) -> bool;
    // TODO Remove this.
    fn is_result(&self) -> bool;
    fn add_arguments(&self, name: &Ident) -> TokenStream;
}

#[derive(Debug, Clone)]
pub struct OperationField<'a> {
    pub(crate) name: &'a str,
    pub(crate) plural_identifier: String,
    pub(crate) sanitized_name: Ident,
    pub(crate) kind: FieldKind<'a>,
}

impl OperationFieldV2 for OperationField<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn plural_identifier(&self) -> &str {
        &self.plural_identifier
    }

    fn sanitized_name(&self) -> &Ident {
        &self.sanitized_name
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
                if constraint.has_unfixed() && !constraint.is_optional() {
                    quote! { #name }
                } else {
                    quote! { &[#name] }
                }
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
    }
}

impl<'a> OperationField<'a> {
    fn new(name: &'a str, kind: FieldKind<'a>) -> Result<Self, Error> {
        Ok(Self {
            name,
            plural_identifier: match kind {
                FieldKind::Element { kind, .. } => format!("{}s", kind.as_str()),
                FieldKind::Successor { .. } => "successors".into(),
                FieldKind::Region { .. } => "regions".into(),
            },
            sanitized_name: sanitize_snake_case_name(name)?,
            kind,
        })
    }

    pub fn new_region(
        name: &'a str,
        constraint: RegionConstraint<'a>,
        sequence_info: SequenceInfo,
    ) -> Result<Self, Error> {
        Self::new(
            name,
            FieldKind::Region {
                constraint,
                sequence_info,
            },
        )
    }

    pub fn new_successor(
        name: &'a str,
        constraint: SuccessorConstraint<'a>,
        sequence_info: SequenceInfo,
    ) -> Result<Self, Error> {
        Self::new(
            name,
            FieldKind::Successor {
                constraint,
                sequence_info,
            },
        )
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
