use crate::{
    context::Context, logical_result::LogicalResult, module::Module,
    operation_pass_manager::OperationPassManager, pass::Pass, string_ref::StringRef,
};
use mlir_sys::{
    mlirPassManagerAddOwnedPass, mlirPassManagerCreate, mlirPassManagerGetNestedUnder,
    mlirPassManagerRun, MlirPassManager,
};
use std::marker::PhantomData;

pub struct PassManager<'c> {
    raw: MlirPassManager,
    _context: PhantomData<&'c Context>,
}

impl<'c> PassManager<'c> {
    pub fn new(context: &Context) -> Self {
        Self {
            raw: unsafe { mlirPassManagerCreate(context.to_raw()) },
            _context: Default::default(),
        }
    }

    pub fn add_pass(&mut self, pass: Pass) {
        unsafe { mlirPassManagerAddOwnedPass(self.raw, pass.to_raw()) }
    }

    pub fn nested_under(&mut self, name: &str) -> OperationPassManager {
        unsafe {
            OperationPassManager::from_raw(mlirPassManagerGetNestedUnder(
                self.raw,
                StringRef::from(name).to_raw(),
            ))
        }
    }

    pub fn run(&self, module: &mut Module) -> LogicalResult {
        LogicalResult::from_raw(unsafe { mlirPassManagerRun(self.raw, module.to_raw()) })
    }
}
