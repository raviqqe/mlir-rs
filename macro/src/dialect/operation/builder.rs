mod type_state_item;
mod type_state_list;

use self::type_state_list::TypeStateList;
use super::{Operation, OperationField};
use quote::format_ident;
use syn::Ident;

pub struct OperationBuilder<'a> {
    operation: &'a Operation<'a>,
    identifier: Ident,
    type_state: TypeStateList,
}

impl<'a> OperationBuilder<'a> {
    pub fn new(operation: &'a Operation<'a>) -> Self {
        Self {
            operation,
            identifier: format_ident!("{}Builder", operation.name()),
            type_state: Self::create_type_state(operation),
        }
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

    fn create_type_state(operation: &Operation) -> TypeStateList {
        TypeStateList::new(
            Self::build_names(operation.required_results()),
            Self::build_names(operation.required_operands()),
            Self::build_names(operation.required_regions()),
            Self::build_names(operation.required_successors()),
            Self::build_names(operation.required_attributes()),
        )
    }

    fn build_names<'b>(
        fields: impl Iterator<Item = &'b (impl OperationField + 'b)>,
    ) -> Vec<String> {
        fields.map(|field| field.name().into()).collect()
    }
}
