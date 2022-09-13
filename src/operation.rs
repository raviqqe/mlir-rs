use crate::{
    context::{Context, ContextRef},
    operation_state::OperationState,
    region::RegionRef,
    string_ref::StringRef,
    value::Value,
};
use core::fmt;
use mlir_sys::{
    mlirOperationCreate, mlirOperationDestroy, mlirOperationDump, mlirOperationGetContext,
    mlirOperationGetNextInBlock, mlirOperationGetNumRegions, mlirOperationGetNumResults,
    mlirOperationGetRegion, mlirOperationGetResult, mlirOperationPrint, mlirOperationVerify,
    MlirOperation, MlirStringRef,
};
use std::{
    ffi::c_void,
    fmt::{Display, Formatter},
    marker::PhantomData,
    mem::{forget, ManuallyDrop},
    ops::Deref,
};

/// An operation.
pub struct Operation<'c> {
    raw: MlirOperation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Operation<'c> {
    /// Creates an operation.
    pub fn new(state: OperationState) -> Self {
        Self {
            raw: unsafe { mlirOperationCreate(&mut state.into_raw()) },
            _context: Default::default(),
        }
    }

    /// Gets a context.
    pub fn context(&self) -> ContextRef {
        unsafe { ContextRef::from_raw(mlirOperationGetContext(self.raw)) }
    }

    /// Gets a result at an index.
    pub fn result(&self, index: usize) -> Option<Value> {
        unsafe {
            if index < mlirOperationGetNumResults(self.raw) as usize {
                Some(Value::from_raw(mlirOperationGetResult(
                    self.raw,
                    index as isize,
                )))
            } else {
                None
            }
        }
    }

    /// Gets a result at an index.
    pub fn region(&self, index: usize) -> Option<RegionRef> {
        unsafe {
            if index < mlirOperationGetNumRegions(self.raw) as usize {
                Some(RegionRef::from_raw(mlirOperationGetRegion(
                    self.raw,
                    index as isize,
                )))
            } else {
                None
            }
        }
    }

    /// Gets the next operation in the same block.
    pub fn next_in_block(&self) -> Option<OperationRef> {
        unsafe {
            let operation = mlirOperationGetNextInBlock(self.raw);

            if operation.ptr.is_null() {
                None
            } else {
                Some(OperationRef::from_raw(operation))
            }
        }
    }

    /// Verifies an operation.
    pub fn verify(&self) -> bool {
        unsafe { mlirOperationVerify(self.raw) }
    }

    /// Dumps an operation.
    pub fn dump(&self) {
        unsafe { mlirOperationDump(self.raw) }
    }

    pub(crate) unsafe fn from_raw(operation: MlirOperation) -> Self {
        Self {
            raw: operation,
            _context: Default::default(),
        }
    }

    pub(crate) unsafe fn into_raw(self) -> MlirOperation {
        let operation = self.raw;

        forget(self);

        operation
    }
}

impl<'c> Drop for Operation<'c> {
    fn drop(&mut self) {
        unsafe { mlirOperationDestroy(self.raw) };
    }
}

impl<'c> Display for &Operation<'c> {
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
            mlirOperationPrint(self.raw, Some(callback), &mut data as *mut _ as *mut c_void);
        }

        data.1
    }
}

/// A reference to an operation.
// TODO Should we split context lifetimes? Or, is it transitively proven that
// 'c > 'a?
pub struct OperationRef<'a> {
    operation: ManuallyDrop<Operation<'a>>,
    _reference: PhantomData<&'a Operation<'a>>,
}

impl<'a> OperationRef<'a> {
    pub(crate) unsafe fn from_raw(operation: MlirOperation) -> Self {
        Self {
            operation: ManuallyDrop::new(Operation::from_raw(operation)),
            _reference: Default::default(),
        }
    }
}

impl<'a> Deref for OperationRef<'a> {
    type Target = Operation<'a>;

    fn deref(&self) -> &Self::Target {
        &self.operation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::Context, location::Location};

    #[test]
    fn new() {
        Operation::new(OperationState::new(
            "foo",
            Location::unknown(&Context::new()),
        ));
    }

    #[test]
    fn result_none() {
        assert!(Operation::new(OperationState::new(
            "foo",
            Location::unknown(&Context::new()),
        ))
        .result(0)
        .is_none());
    }

    #[test]
    fn region_none() {
        assert!(Operation::new(OperationState::new(
            "foo",
            Location::unknown(&Context::new()),
        ))
        .region(0)
        .is_none());
    }
}
