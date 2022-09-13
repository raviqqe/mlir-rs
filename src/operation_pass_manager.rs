use crate::{pass::Pass, pass_manager::PassManager, string_ref::StringRef};
use mlir_sys::{mlirOpPassManagerAddOwnedPass, mlirOpPassManagerGetNestedUnder, MlirOpPassManager};
use std::marker::PhantomData;

/// An operation pass manager.
pub struct OperationPassManager<'a> {
    raw: MlirOpPassManager,
    _parent: PhantomData<&'a PassManager<'a>>,
}

impl<'a> OperationPassManager<'a> {
    /// Gets an operation pass manager for nested operations corresponding to a given name.
    pub fn nested_under(&mut self, name: &str) -> OperationPassManager {
        unsafe {
            Self::from_raw(mlirOpPassManagerGetNestedUnder(
                self.raw,
                StringRef::from(name).to_raw(),
            ))
        }
    }

    /// Adds a pass to an operation pass manager.
    pub fn add_pass(&mut self, pass: Pass) {
        unsafe { mlirOpPassManagerAddOwnedPass(self.raw, pass.to_raw()) }
    }

    pub(crate) unsafe fn from_raw(raw: MlirOpPassManager) -> Self {
        Self {
            raw,
            _parent: Default::default(),
        }
    }
}
