use quote::{format_ident, quote};
use syn::GenericArgument;

#[derive(Debug)]
pub struct TypeStateItem {
    field_name: String,
    generic_parameter: GenericArgument,
}

impl TypeStateItem {
    pub fn new(index: usize, field_name: String) -> Self {
        let identifier = format_ident!("T{}", index);

        Self {
            generic_parameter: syn::parse2(quote!(#identifier)).expect("valid GenericArgument"),
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
