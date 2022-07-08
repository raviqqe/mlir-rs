use crate::{
    attribute::Attribute,
    context::Context,
    identifier::Identifier,
    location::Location,
    utility::{as_string_ref, into_raw_array},
};
use mlir_sys::{
    mlirNamedAttributeGet, mlirOperationStateAddAttributes, mlirOperationStateGet,
    MlirOperationState,
};
use std::marker::PhantomData;

pub struct OperationState<'c> {
    state: MlirOperationState,
    _context: PhantomData<&'c Context>,
}

impl<'c> OperationState<'c> {
    pub fn new(name: impl AsRef<str>, location: Location<'c>) -> Self {
        Self {
            state: unsafe {
                mlirOperationStateGet(as_string_ref(name.as_ref()), location.to_raw())
            },
            _context: Default::default(),
        }
    }

    pub fn add_attributes(&mut self, attributes: Vec<(Identifier, Attribute<'c>)>) -> &mut Self {
        unsafe {
            mlirOperationStateAddAttributes(
                &mut self.state,
                attributes.len() as isize,
                into_raw_array(
                    attributes
                        .into_iter()
                        .map(|(identifier, attribute)| {
                            mlirNamedAttributeGet(identifier.to_raw(), attribute.to_raw())
                        })
                        .collect(),
                ),
            )
        }

        self
    }

    pub(crate) unsafe fn into_raw(self) -> MlirOperationState {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::operation::Operation;

    #[test]
    fn new() {
        let context = Context::new();
        let mut state = OperationState::new("foo", Location::unknown(&context));

        state.add_attributes(vec![(
            Identifier::new(&context, "bar"),
            Attribute::parse(&context, "unit"),
        )]);

        Operation::new(state);
    }
}
