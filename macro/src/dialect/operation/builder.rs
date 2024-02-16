mod type_state_item;
mod type_state_list;

use self::{type_state_item::TypeStateItem, type_state_list::TypeStateList};
use super::{super::error::Error, Operation};
use quote::format_ident;
use syn::Ident;

pub struct OperationBuilder<'o> {
    operation: &'o Operation<'o>,
    identifier: Ident,
    type_state: TypeStateList,
}

impl<'o> OperationBuilder<'o> {
    pub fn new(operation: &'o Operation<'o>) -> Result<Self, Error> {
        Ok(Self {
            operation,
            identifier: format_ident!("{}Builder", operation.class_name()?),
            type_state: Self::create_type_state(operation),
        })
    }

    pub fn operation(&self) -> &Operation {
        self.operation
    }

    pub fn identifier(&self) -> &Ident {
        &self.identifier
    }

    pub fn type_state(&self) -> &TypeStateList {
        &self.type_state
    }

    fn create_type_state(operation: &'o Operation<'o>) -> TypeStateList {
        TypeStateList::new(
            operation
                .required_fields()
                .enumerate()
                .map(|(index, field)| TypeStateItem::new(index, field.name().to_string()))
                .collect(),
        )
    }
}
