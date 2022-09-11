use crate::{
    context::{Context, ContextRef},
    operation_state::OperationState,
    region::RegionRef,
    string_ref::StringRef,
    value::Value,
};
use mlir_sys::{
    mlirOperationCreate, mlirOperationDestroy, mlirOperationDump, mlirOperationGetContext,
    mlirOperationGetNextInBlock, mlirOperationGetRegion, mlirOperationGetResult,
    mlirOperationPrint, mlirOperationVerify, MlirOperation, MlirStringRef,
};
use std::{
    ffi::c_void,
    marker::PhantomData,
    mem::{forget, ManuallyDrop},
    ops::Deref,
};

pub struct Operation<'c> {
    operation: MlirOperation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Operation<'c> {
    pub fn new(state: OperationState) -> Self {
        Self {
            operation: unsafe { mlirOperationCreate(&mut state.into_raw()) },
            _context: Default::default(),
        }
    }

    pub fn context(&self) -> ContextRef {
        unsafe { ContextRef::from_raw(mlirOperationGetContext(self.operation)) }
    }

    pub fn result(&self, index: usize) -> Value {
        unsafe { Value::from_raw(mlirOperationGetResult(self.operation, index as isize)) }
    }

    pub fn region(&self, index: usize) -> RegionRef {
        unsafe { RegionRef::from_raw(mlirOperationGetRegion(self.operation, index as isize)) }
    }

    pub fn next_in_block(&self) -> Option<OperationRef> {
        unsafe {
            let operation = mlirOperationGetNextInBlock(self.operation);

            if operation.ptr.is_null() {
                None
            } else {
                Some(OperationRef::from_raw(operation))
            }
        }
    }

    pub fn verify(&self) -> bool {
        unsafe { mlirOperationVerify(self.operation) }
    }

    pub fn print(&self) -> String {
        let mut strings: Vec<StringRef> = vec![];

        unsafe extern "C" fn callback(string: MlirStringRef, data: *mut c_void) {
            (&mut *(data as *mut Vec<StringRef>)).push(StringRef::from_raw(string));
        }

        unsafe {
            mlirOperationPrint(
                self.operation,
                Some(callback),
                &mut strings as *mut _ as *mut c_void,
            );
        }

        strings
            .iter()
            .map(|string| string.as_str())
            .collect::<Vec<&str>>()
            .concat()
    }

    pub fn dump(&self) {
        unsafe { mlirOperationDump(self.operation) }
    }

    pub(crate) unsafe fn from_raw(operation: MlirOperation) -> Self {
        Self {
            operation,
            _context: Default::default(),
        }
    }

    pub(crate) unsafe fn into_raw(self) -> MlirOperation {
        let operation = self.operation;

        forget(self);

        operation
    }
}

impl<'c> Drop for Operation<'c> {
    fn drop(&mut self) {
        unsafe { mlirOperationDestroy(self.operation) };
    }
}

// TODO Should we split context lifetimes? Or, is it transitively proven that 'c > 'a?
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
}
