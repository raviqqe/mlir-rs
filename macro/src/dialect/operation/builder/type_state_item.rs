use quote::{format_ident, quote};
use syn::GenericArgument;

#[derive(Debug)]
pub struct TypeStateItem {
    field_name: String,
    generic_param: GenericArgument,
}

impl TypeStateItem {
    pub fn new(index: usize, field_name: String) -> Self {
        Self {
            generic_param: {
                let ident = format_ident!("T{}", index);
                syn::parse2(quote!(#ident)).expect("Ident is a valid GenericArgument")
            },
            field_name,
        }
    }

    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    pub fn generic_param(&self) -> &GenericArgument {
        &self.generic_param
    }
}
