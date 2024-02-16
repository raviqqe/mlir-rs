use super::{element_kind::ElementKind, SequenceInfo, VariadicKind};
use crate::dialect::{
    types::{SuccessorConstraint, TypeConstraint},
    utility::{generate_iterator_type, generate_result_type},
};
use syn::{parse_quote, Type};

#[derive(Debug, Clone)]
pub enum FieldKind<'a> {
    Element {
        kind: ElementKind,
        constraint: TypeConstraint<'a>,
        sequence_info: SequenceInfo,
        variadic_kind: VariadicKind,
    },
    Successor {
        constraint: SuccessorConstraint<'a>,
        sequence_info: SequenceInfo,
    },
}

impl<'a> FieldKind<'a> {
    pub fn is_optional(&self) -> bool {
        match self {
            Self::Element { constraint, .. } => constraint.is_optional(),
            Self::Successor { .. } => false,
        }
    }

    pub fn parameter_type(&self) -> Type {
        match self {
            Self::Element {
                kind, constraint, ..
            } => {
                let base_type: Type = match kind {
                    ElementKind::Operand => {
                        parse_quote!(::melior::ir::Value<'c, '_>)
                    }
                    ElementKind::Result => {
                        parse_quote!(::melior::ir::Type<'c>)
                    }
                };
                if constraint.is_variadic() {
                    parse_quote! { &[#base_type] }
                } else {
                    base_type
                }
            }
            Self::Successor { constraint, .. } => {
                let r#type: Type = parse_quote!(&::melior::ir::Block<'c>);
                if constraint.is_variadic() {
                    parse_quote!(&[#r#type])
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
                let base_type: Type = match kind {
                    ElementKind::Operand => {
                        parse_quote!(::melior::ir::Value<'c, '_>)
                    }
                    ElementKind::Result => {
                        parse_quote!(::melior::ir::operation::OperationResult<'c, '_>)
                    }
                };

                if !constraint.is_variadic() {
                    generate_result_type(base_type)
                } else if variadic_kind == &VariadicKind::AttributeSized {
                    generate_result_type(generate_iterator_type(base_type))
                } else {
                    generate_iterator_type(base_type)
                }
            }
            Self::Successor { constraint, .. } => {
                let r#type: Type = parse_quote!(::melior::ir::BlockRef<'c, '_>);
                if constraint.is_variadic() {
                    generate_iterator_type(r#type)
                } else {
                    generate_result_type(r#type)
                }
            }
        }
    }
}
