use crate::{
    attribute::Attribute,
    location::Location,
    region::Region,
    utility::{self, into_raw_array},
    value::Value,
};
use mlir_sys::MlirOperationState;

pub struct OperationState<'c> {
    name: String,
    location: Location<'c>,
    results: Vec<Type>,
    operands: Vec<Value>,
    regions: Vec<Region>,
    successors: Vec<Block>,
    attributes: Vec<Attribute<'c>>,
    enable_result_type_inference: bool,
}

impl<'c> OperationState<'c> {
    pub fn new(name: impl Into<String>, location: Location) -> Self {
        Self {
            name: name.into(),
            location,
            results: vec![],
            operands: vec![],
            regions: vec![],
            successors: vec![],
            attributes: vec![],
            enable_result_type_inference: false,
        }
    }

    pub(crate) fn into_raw(self) -> MlirOperationState {
        unsafe {
            MlirOperationState {
                name: utility::as_string_ref(&self.name),
                location: self.location.to_raw(),
                nResults: self.results.len(),
                results: self.results,
                nOperands: self.operands.len() as isize,
                operands: into_raw_array(
                    self.operands
                        .into_iter()
                        .map(|value| value.to_raw())
                        .collect::<Vec<_>>(),
                ),
                nRegions: self.regions.len() as isize,
                regions: into_raw_array(
                    self.regions
                        .into_iter()
                        .map(|region| region.to_raw())
                        .collect::<Vec<_>>(),
                ),
                nSuccessors: self.successors.len(),
                successors: into_raw_array(
                    self.successors
                        .into_iter()
                        .map(|block| block.to_raw())
                        .collect::<Vec<_>>(),
                ),
                nAttributes: self.attributes.len() as isize,
                attributes: into_raw_array(
                    self.attributes
                        .into_iter()
                        .map(|attribute| attribute.to_raw())
                        .collect::<Vec<_>>(),
                ),
                enableResultTypeInference: self.enable_result_type_inference,
            }
        }
    }
}
