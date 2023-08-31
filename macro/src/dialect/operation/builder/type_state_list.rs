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

    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.items.iter().map(|item| item.field_name())
    }

    pub fn parameters(&self) -> impl Iterator<Item = &GenericArgument> {
        self.items.iter().map(|item| item.generic_param())
    }

    pub fn parameters_without<'a>(
        &'a self,
        field_name: &'a str,
    ) -> impl Iterator<Item = &GenericArgument> {
        self.items
            .iter()
            .filter(move |item| item.field_name() != field_name)
            .map(|item| item.generic_param())
    }

    pub fn arguments_set<'a>(
        &'a self,
        field_name: &'a str,
        set: bool,
    ) -> impl Iterator<Item = &GenericArgument> {
        self.items.iter().map(move |item| {
            if item.field_name() == field_name {
                self.set_argument(set)
            } else {
                item.generic_param()
            }
        })
    }

    pub fn arguments_all_set(&self, set: bool) -> impl Iterator<Item = &GenericArgument> {
        repeat(self.set_argument(set)).take(self.items.len())
    }

    fn set_argument(&self, set: bool) -> &GenericArgument {
        if set {
            &self.set
        } else {
            &self.unset
        }
    }
}
