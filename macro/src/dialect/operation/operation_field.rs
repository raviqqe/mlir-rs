use super::{element_kind::ElementKind, field_kind::FieldKind, SequenceInfo, VariadicKind};
use crate::dialect::{
    error::Error,
    types::{AttributeConstraint, RegionConstraint, SuccessorConstraint, TypeConstraint},
    utility::sanitize_snake_case_name,
};
use proc_macro2::Ident;

pub trait OperationFieldV2 {
    fn name(&self) -> &str;
    fn sanitized_name(&self) -> &Ident;
}

#[derive(Debug, Clone)]
pub struct OperationField<'a> {
    pub(crate) name: &'a str,
    pub(crate) sanitized_name: Ident,
    pub(crate) kind: FieldKind<'a>,
}

impl<'a> OperationField<'a> {
    fn new(name: &'a str, kind: FieldKind<'a>) -> Result<Self, Error> {
        Ok(Self {
            name,
            sanitized_name: sanitize_snake_case_name(name)?,
            kind,
        })
    }

    pub fn new_attribute(
        name: &'a str,
        constraint: AttributeConstraint<'a>,
    ) -> Result<Self, Error> {
        Self::new(name, FieldKind::Attribute { constraint })
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
