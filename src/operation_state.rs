use crate::{
    attribute::Attribute,
    block::Block,
    context::Context,
    identifier::Identifier,
    location::Location,
    r#type::Type,
    region::Region,
    utility::{as_string_ref, into_raw_array},
    value::Value,
};
use mlir_sys::{
    mlirNamedAttributeGet, mlirOperationStateAddAttributes, mlirOperationStateAddOperands,
    mlirOperationStateAddOwnedRegions, mlirOperationStateAddResults,
    mlirOperationStateAddSuccessors, mlirOperationStateGet, MlirOperationState,
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

    pub fn add_results(&mut self, results: Vec<Type<'c>>) -> &mut Self {
        unsafe {
            mlirOperationStateAddResults(
                &mut self.state,
                results.len() as isize,
                into_raw_array(results.iter().map(|r#type| r#type.to_raw()).collect()),
            )
        }

        self
    }

    pub fn add_operands(&mut self, operandss: Vec<Value>) -> &mut Self {
        unsafe {
            mlirOperationStateAddOperands(
                &mut self.state,
                operandss.len() as isize,
                into_raw_array(operandss.iter().map(|value| value.to_raw()).collect()),
            )
        }

        self
    }

    pub fn add_owned_regions(&mut self, regionss: Vec<Region>) -> &mut Self {
        unsafe {
            mlirOperationStateAddOwnedRegions(
                &mut self.state,
                regionss.len() as isize,
                into_raw_array(regionss.iter().map(|region| region.to_raw()).collect()),
            )
        }

        self
    }

    pub fn add_successors(&mut self, successorss: Vec<Block>) -> &mut Self {
        unsafe {
            mlirOperationStateAddSuccessors(
                &mut self.state,
                successorss.len() as isize,
                into_raw_array(successorss.iter().map(|block| block.to_raw()).collect()),
            )
        }

        self
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
        Operation::new(OperationState::new(
            "foo",
            Location::unknown(&Context::new()),
        ));
    }

    #[test]
    fn add_results() {
        let context = Context::new();
        let mut state = OperationState::new("foo", Location::unknown(&context));

        state.add_results(vec![Type::parse(&context, "i32")]);

        Operation::new(state);
    }

    #[test]
    fn add_owned_regions() {
        let context = Context::new();
        let mut state = OperationState::new("foo", Location::unknown(&context));

        state.add_owned_regions(vec![Region::new()]);

        Operation::new(state);
    }

    #[test]
    fn add_successors() {
        let context = Context::new();
        let mut state = OperationState::new("foo", Location::unknown(&context));

        state.add_successors(vec![Block::new(vec![])]);

        Operation::new(state);
    }

    #[test]
    fn add_attributes() {
        let context = Context::new();
        let mut state = OperationState::new("foo", Location::unknown(&context));

        state.add_attributes(vec![(
            Identifier::new(&context, "bar"),
            Attribute::parse(&context, "unit"),
        )]);

        Operation::new(state);
    }
}
