use crate::dialect::{
    error::Error,
    operation::operation_field::OperationFieldLike,
    types::AttributeConstraint,
    utility::{generate_result_type, sanitize_snake_case_name},
};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{parse_quote, Type};

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

    pub fn is_optional(&self) -> bool {
        self.constraint.is_optional()
    }

    pub fn is_unit(&self) -> bool {
        self.constraint.is_unit()
    }
}

impl OperationFieldLike for Attribute<'_> {
    fn name(&self) -> &str {
        &self.name
    }

    fn plural_identifier(&self) -> &str {
        "attributes"
    }

    fn sanitized_name(&self) -> &Ident {
        &self.sanitized_name
    }

    fn parameter_type(&self) -> Type {
        if self.constraint.is_unit() {
            parse_quote!(bool)
        } else {
            let r#type = self.constraint.storage_type();
            parse_quote!(#r#type<'c>)
        }
    }

    fn return_type(&self) -> Type {
        if self.constraint.is_unit() {
            parse_quote!(bool)
        } else {
            generate_result_type(self.parameter_type())
        }
    }

    fn is_optional(&self) -> bool {
        self.is_optional() || self.constraint.has_default_value()
    }

    fn is_result(&self) -> bool {
        false
    }

    fn add_arguments(&self, name: &Ident) -> TokenStream {
        let name_string = &self.name;

        quote! {
            &[(
                ::melior::ir::Identifier::new(self.context, #name_string),
                #name.into(),
            )]
        }
    }
}
