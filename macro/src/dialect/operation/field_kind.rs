use super::{element_kind::ElementKind, SequenceInfo, VariadicKind};
use crate::dialect::types::{
    AttributeConstraint, RegionConstraint, SuccessorConstraint, TypeConstraint,
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
    Attribute {
        constraint: AttributeConstraint<'a>,
    },
    Successor {
        constraint: SuccessorConstraint<'a>,
        sequence_info: SequenceInfo,
    },
    Region {
        constraint: RegionConstraint<'a>,
        sequence_info: SequenceInfo,
    },
}

impl<'a> FieldKind<'a> {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Element { kind, .. } => kind.as_str(),
            Self::Attribute { .. } => "attribute",
            Self::Successor { .. } => "successor",
            Self::Region { .. } => "region",
        }
    }

    pub fn is_optional(&self) -> bool {
        match self {
            Self::Element { constraint, .. } => constraint.is_optional(),
            Self::Attribute { constraint, .. } => {
                constraint.is_optional() || constraint.has_default_value()
            }
            Self::Successor { .. } | Self::Region { .. } => false,
        }
    }

    pub fn is_result(&self) -> bool {
        matches!(
            self,
            Self::Element {
                kind: ElementKind::Result,
                ..
            }
        )
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
            Self::Attribute { constraint } => {
                if constraint.is_unit() {
                    parse_quote!(bool)
                } else {
                    let r#type = constraint.storage_type();
                    parse_quote!(#r#type<'c>)
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
            Self::Region { constraint, .. } => {
                let r#type: Type = parse_quote!(::melior::ir::Region<'c>);
                if constraint.is_variadic() {
                    parse_quote!(Vec<#r#type>)
                } else {
                    r#type
                }
            }
        }
    }

    fn create_result_type(r#type: Type) -> Type {
        parse_quote!(Result<#r#type, ::melior::Error>)
    }

    fn create_iterator_type(r#type: Type) -> Type {
        parse_quote!(impl Iterator<Item = #r#type>)
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
                    Self::create_result_type(base_type)
                } else if variadic_kind == &VariadicKind::AttributeSized {
                    Self::create_result_type(Self::create_iterator_type(base_type))
                } else {
                    Self::create_iterator_type(base_type)
                }
            }
            Self::Attribute { constraint } => {
                if constraint.is_unit() {
                    parse_quote!(bool)
                } else {
                    Self::create_result_type(self.parameter_type())
                }
            }
            Self::Successor { constraint, .. } => {
                let r#type: Type = parse_quote!(::melior::ir::BlockRef<'c, '_>);
                if constraint.is_variadic() {
                    Self::create_iterator_type(r#type)
                } else {
                    Self::create_result_type(r#type)
                }
            }
            Self::Region { constraint, .. } => {
                let r#type: Type = parse_quote!(::melior::ir::RegionRef<'c, '_>);
                if constraint.is_variadic() {
                    Self::create_iterator_type(r#type)
                } else {
                    Self::create_result_type(r#type)
                }
            }
        }
    }
}
