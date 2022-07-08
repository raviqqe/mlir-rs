use crate::{
    attribute::Attribute,
    block::Block,
    location::Location,
    r#type::Type,
    region::Region,
    utility::{as_string_ref, into_raw_array},
    value::Value,
};
use mlir_sys::{mlirIdentifierGet, mlirNamedAttributeGet, MlirOperationState};
use std::collections::HashMap;

pub struct OperationState<'c> {
    name: String,
    location: Location<'c>,
    results: Vec<Type<'c>>,
    operands: Vec<Value>,
    regions: Vec<Region>,
    successors: Vec<Block>,
    attributes: HashMap<String, Attribute<'c>>,
    enable_result_type_inference: bool,
}

impl<'c> OperationState<'c> {
    pub fn new(name: impl Into<String>, location: Location<'c>) -> Self {
        Self {
            name: name.into(),
            location,
            results: vec![],
            operands: vec![],
            regions: vec![],
            successors: vec![],
            attributes: Default::default(),
            enable_result_type_inference: false,
        }
    }

    pub fn set_attribute(&mut self, name: impl AsRef<str>, attribute: Attribute<'c>) {
        self.attributes.insert(name.as_ref().into(), attribute);
    }

    pub(crate) fn into_raw(self) -> MlirOperationState {
        unsafe {
            MlirOperationState {
                name: as_string_ref(&self.name),
                location: self.location.to_raw(),
                nResults: self.results.len() as isize,
                results: into_raw_array(
                    self.results
                        .into_iter()
                        .map(|r#type| r#type.to_raw())
                        .collect(),
                ),
                nOperands: self.operands.len() as isize,
                operands: into_raw_array(
                    self.operands
                        .into_iter()
                        .map(|value| value.to_raw())
                        .collect(),
                ),
                nRegions: self.regions.len() as isize,
                regions: into_raw_array(
                    self.regions
                        .into_iter()
                        .map(|region| region.to_raw())
                        .collect(),
                ),
                nSuccessors: self.successors.len() as isize,
                successors: into_raw_array(
                    self.successors
                        .into_iter()
                        .map(|block| block.to_raw())
                        .collect(),
                ),
                nAttributes: self.attributes.len() as isize,
                attributes: into_raw_array(
                    self.attributes
                        .into_iter()
                        .map(|(name, attribute)| {
                            mlirNamedAttributeGet(
                                mlirIdentifierGet(
                                    self.location.context().to_raw(),
                                    as_string_ref(&name),
                                ),
                                attribute.to_raw(),
                            )
                        })
                        .collect(),
                ),
                enableResultTypeInference: self.enable_result_type_inference,
            }
        }
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

        state.set_attribute("bar", Attribute::parse(&context, "unit"));

        Operation::new(state);
    }
}
