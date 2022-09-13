use crate::{
    context::Context, logical_result::LogicalResult, module::Module,
    operation_pass_manager::OperationPassManager, pass::Pass, string_ref::StringRef,
};
use mlir_sys::{
    mlirPassManagerAddOwnedPass, mlirPassManagerCreate, mlirPassManagerDestroy,
    mlirPassManagerGetAsOpPassManager, mlirPassManagerGetNestedUnder, mlirPassManagerRun,
    MlirPassManager,
};
use std::marker::PhantomData;

/// A pass manager.
pub struct PassManager<'c> {
    raw: MlirPassManager,
    _context: PhantomData<&'c Context>,
}

impl<'c> PassManager<'c> {
    /// Creates a pass manager.
    pub fn new(context: &Context) -> Self {
        Self {
            raw: unsafe { mlirPassManagerCreate(context.to_raw()) },
            _context: Default::default(),
        }
    }

    /// Gets an operation pass manager for nested operations corresponding to a
    /// given name.
    pub fn nested_under(&self, name: &str) -> OperationPassManager {
        unsafe {
            OperationPassManager::from_raw(mlirPassManagerGetNestedUnder(
                self.raw,
                StringRef::from(name).to_raw(),
            ))
        }
    }

    /// Adds a pass.
    pub fn add_pass(&self, pass: Pass) {
        unsafe { mlirPassManagerAddOwnedPass(self.raw, pass.to_raw()) }
    }

    /// Runs passes added to a pass manager against a module.
    pub fn run(&self, module: &mut Module) -> LogicalResult {
        LogicalResult::from_raw(unsafe { mlirPassManagerRun(self.raw, module.to_raw()) })
    }

    /// Converts a pass manager to an operation pass manager.
    pub fn as_operation_pass_manager(&mut self) -> OperationPassManager {
        unsafe { OperationPassManager::from_raw(mlirPassManagerGetAsOpPassManager(self.raw)) }
    }
}

impl<'c> Drop for PassManager<'c> {
    fn drop(&mut self) {
        unsafe { mlirPassManagerDestroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::location::Location;

    #[test]
    fn new() {
        let context = Context::new();

        PassManager::new(&context);
    }

    #[test]
    fn add_pass() {
        let context = Context::new();

        PassManager::new(&context).add_pass(Pass::convert_func_to_llvm());
    }

    #[test]
    fn run() {
        let context = Context::new();
        let mut manager = PassManager::new(&context);

        manager.add_pass(Pass::convert_func_to_llvm());
        manager.run(&mut Module::new(Location::unknown(&context)));
    }
}
