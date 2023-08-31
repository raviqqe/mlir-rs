use super::type_state_item::TypeStateItem;
use quote::quote;
use std::iter::repeat;
use syn::GenericArgument;

#[derive(Debug)]
pub struct TypeStateList {
    items: Vec<TypeStateItem>,
    unset: GenericArgument,
    set: GenericArgument,
}

impl TypeStateList {
    pub fn new(items: Vec<TypeStateItem>) -> Self {
        Self {
            items,
            unset: syn::parse2(quote!(::melior::dialect::ods::__private::Unset)).unwrap(),
            set: syn::parse2(quote!(::melior::dialect::ods::__private::Set)).unwrap(),
        }
    }

    pub fn items(&self) -> impl Iterator<Item = &TypeStateItem> {
        self.items.iter()
    }

    pub fn unset(&self) -> &GenericArgument {
        &self.unset
    }

    pub fn set(&self) -> &GenericArgument {
        &self.set
    }

    pub fn parameters(&self) -> impl Iterator<Item = &GenericArgument> {
        self.items().map(|item| item.generic_param())
    }

    pub fn parameters_without<'a>(
        &'a self,
        field_name: &'a str,
    ) -> impl Iterator<Item = &GenericArgument> + '_ {
        self.items()
            .filter(move |item| item.field_name() != field_name)
            .map(|item| item.generic_param())
    }

    pub fn arguments_replace<'a>(
        &'a self,
        field_name: &'a str,
        argument: &'a GenericArgument,
    ) -> impl Iterator<Item = &GenericArgument> + '_ {
        self.items().map(move |item| {
            if item.field_name() == field_name {
                argument
            } else {
                item.generic_param()
            }
        })
    }

    pub fn arguments_all<'a>(
        &'a self,
        argument: &'a GenericArgument,
    ) -> impl Iterator<Item = &GenericArgument> + '_ {
        repeat(argument).take(self.items.len())
    }
}
