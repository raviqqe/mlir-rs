use crate::dialect::operation::operation_field::OperationFieldV2;
use crate::dialect::types::AttributeConstraint;
use crate::dialect::utility::generate_result_type;
use crate::dialect::{error::Error, utility::sanitize_snake_case_name};
use syn::{parse_quote, Ident, Type};

#[derive(Debug)]
pub struct Attribute<'a> {
    name: &'a str,
    sanitized_name: Ident,
    constraint: AttributeConstraint<'a>,
}

impl<'a> Attribute<'a> {
    pub fn new(name: &'a str, constraint: AttributeConstraint<'a>) -> Result<Self, Error> {
        Ok(Self {
            name,
            sanitized_name: sanitize_snake_case_name(name)?,
            constraint,
        })
    }

    pub fn constraint(&self) -> &AttributeConstraint {
        &self.constraint
    }

    pub fn parameter_type(&self) -> Type {
        if self.constraint().is_unit() {
            parse_quote!(bool)
        } else {
            let r#type = self.constraint().storage_type();
            parse_quote!(#r#type<'c>)
        }
    }

    pub fn return_type(&self) -> Type {
        if self.constraint.is_unit() {
            parse_quote!(bool)
        } else {
            generate_result_type(self.parameter_type())
        }
    }
}

impl OperationFieldV2 for Attribute<'_> {
    fn name(&self) -> &str {
        &self.name
    }

    fn sanitized_name(&self) -> &Ident {
        &self.sanitized_name
    }
}
