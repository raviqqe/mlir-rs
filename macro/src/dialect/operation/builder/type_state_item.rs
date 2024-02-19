use quote::format_ident;
use syn::{parse_quote, GenericArgument};

#[derive(Debug)]
pub struct TypeStateItem {
    field_name: String,
    generic_parameter: GenericArgument,
}

impl TypeStateItem {
    pub fn new(field_name: String, prefix: &'static str, index: usize) -> Self {
        let identifier = format_ident!("{prefix}{index}");

        Self {
            generic_parameter: parse_quote!(#identifier),
            field_name,
        }
    }

    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    pub fn generic_parameter(&self) -> &GenericArgument {
        &self.generic_parameter
    }
}
