use super::operation_field::OperationFieldV2;
use crate::dialect::error::Error;
use crate::dialect::types::AttributeConstraint;
use crate::dialect::utility::sanitize_snake_case_name;
use syn::Ident;

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
}

impl OperationFieldV2 for Attribute<'_> {
    fn name(&self) -> &str {
        &self.name
    }

    fn sanitized_name(&self) -> &Ident {
        &self.sanitized_name
    }
}
