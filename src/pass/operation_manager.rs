use super::Manager;
use crate::{pass::Pass, string_ref::StringRef};
use mlir_sys::{
    mlirOpPassManagerAddOwnedPass, mlirOpPassManagerGetNestedUnder, mlirPrintPassPipeline,
    MlirOpPassManager, MlirStringRef,
};
use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
};

/// An operation pass manager.
#[derive(Clone, Copy, Debug)]
pub struct OperationManager<'a> {
    raw: MlirOpPassManager,
    _parent: PhantomData<&'a Manager<'a>>,
}

impl<'a> OperationManager<'a> {
    /// Gets an operation pass manager for nested operations corresponding to a
    /// given name.
    pub fn nested_under(&self, name: &str) -> Self {
        unsafe {
            Self::from_raw(mlirOpPassManagerGetNestedUnder(
                self.raw,
                StringRef::from(name).to_raw(),
            ))
        }
    }

    /// Adds a pass.
    pub fn add_pass(&self, pass: Pass) {
        unsafe { mlirOpPassManagerAddOwnedPass(self.raw, pass.to_raw()) }
    }

    pub(crate) unsafe fn to_raw(self) -> MlirOpPassManager {
        self.raw
    }

    pub(crate) unsafe fn from_raw(raw: MlirOpPassManager) -> Self {
        Self {
            raw,
            _parent: Default::default(),
        }
    }
}

impl<'a> Display for OperationManager<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe extern "C" fn callback(string: MlirStringRef, data: *mut c_void) {
            let data = &mut *(data as *mut (&mut Formatter, fmt::Result));
            let result = write!(data.0, "{}", StringRef::from_raw(string).as_str());

            if data.1.is_ok() {
                data.1 = result;
            }
        }

        unsafe {
            mlirPrintPassPipeline(self.raw, Some(callback), &mut data as *mut _ as *mut c_void);
        }

        data.1
    }
}
