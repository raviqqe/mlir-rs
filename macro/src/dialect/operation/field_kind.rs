use super::{SequenceInfo, VariadicKind};
use crate::dialect::{
    types::TypeConstraint,
    utility::{generate_iterator_type, generate_result_type},
};
use syn::{parse_quote, Type};

#[derive(Debug)]
pub enum FieldKind<'a> {
    Element {
        kind: ElementKind,
        constraint: TypeConstraint<'a>,
        sequence_info: SequenceInfo,
        variadic_kind: VariadicKind,
    },
}

impl<'a> FieldKind<'a> {
    pub fn is_optional(&self) -> bool {
        match self {
            Self::Element { constraint, .. } => constraint.is_optional(),
        }
    }

    pub fn parameter_type(&self) -> Type {
        match self {
            Self::Element {
                kind, constraint, ..
            } => {
                let r#type: Type = parse_quote!(::melior::ir::Value<'c, '_>);

                if constraint.is_variadic() {
                    parse_quote! { &[#r#type] }
                } else {
                    r#type
                }
            }
        }
    }

    pub fn return_type(&self) -> Type {
        match self {
            Self::Element {
                kind,
                constraint,
                variadic_kind,
                ..
            } => {
                let r#type: Type = parse_quote!(::melior::ir::Value<'c, '_>);

                if !constraint.is_variadic() {
                    generate_result_type(r#type)
                } else if variadic_kind == &VariadicKind::AttributeSized {
                    generate_result_type(generate_iterator_type(r#type))
                } else {
                    generate_iterator_type(r#type)
                }
            }
        }
    }
}
