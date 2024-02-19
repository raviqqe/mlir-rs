use std::iter::repeat;
use syn::{parse_quote, GenericArgument};

#[derive(Debug)]
pub struct TypeStateList {
    results: Vec<String>,
    operands: Vec<String>,
    regions: Vec<String>,
    successors: Vec<String>,
    attributes: Vec<String>,
}

impl TypeStateList {
    pub fn new(
        results: Vec<String>,
        operands: Vec<String>,
        regions: Vec<String>,
        successors: Vec<String>,
        attributes: Vec<String>,
    ) -> Self {
        Self {
            results,
            operands,
            regions,
            successors,
            attributes,
        }
    }

    pub fn parameters(&self) -> impl Iterator<Item = &GenericArgument> {
        self.items.iter().map(|item| item.generic_parameter())
    }

    pub fn parameters_without<'a>(
        &'a self,
        field_name: &'a str,
    ) -> impl Iterator<Item = &GenericArgument> {
        self.items
            .iter()
            .filter(move |item| item.field_name() != field_name)
            .map(|item| item.generic_parameter())
    }

    pub fn arguments_set<'a>(
        &'a self,
        field_name: &'a str,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        self.items.iter().map(move |item| {
            if item.field_name() == field_name {
                self.set_argument(set)
            } else {
                item.generic_parameter().clone()
            }
        })
    }

    pub fn arguments_all_set(&self, set: bool) -> impl Iterator<Item = GenericArgument> {
        repeat(self.set_argument(set)).take(self.items.len())
    }

    fn set_argument(&self, set: bool) -> GenericArgument {
        if set {
            parse_quote!(::melior::dialect::ods::__private::Set)
        } else {
            parse_quote!(::melior::dialect::ods::__private::Unset)
        }
    }
}
