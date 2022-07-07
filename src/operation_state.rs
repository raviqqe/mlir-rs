use crate::{attribute::Attribute, location::Location, region::Region, utility};
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
                nOperands: self.operands.len(),
                operands: self.operands,
                nRegions: self.regions.len() as isize,
                regions: self.regions,
                nSuccessors: self.successors.len(),
                successors: self.blocks,
                nAttributes: self.attributes.len() as isize,
                attributes: self.attributes,
                enableResultTypeInference: self.enable_result_type_inference,
            }
        }
    }
}
