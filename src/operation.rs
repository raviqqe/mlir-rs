use crate::{
    context::ContextRef, operation_state::OperationState, region::RegionRef, string_ref::StringRef,
    value::Value,
};
use mlir_sys::{
    mlirOperationCreate, mlirOperationDestroy, mlirOperationDump, mlirOperationGetContext,
    mlirOperationGetRegion, mlirOperationGetResult, mlirOperationPrint, MlirOperation,
    MlirStringRef,
};
use std::{ffi::c_void, marker::PhantomData, mem::ManuallyDrop, ops::Deref};

pub struct Operation {
    operation: MlirOperation,
}

impl Operation {
    pub fn new(state: OperationState) -> Self {
        Self {
            operation: unsafe { mlirOperationCreate(&mut state.into_raw()) },
        }
    }

    pub fn context(&self) -> ContextRef {
        unsafe { ContextRef::from_raw(mlirOperationGetContext(self.operation)) }
    }

    pub fn result(&self, index: usize) -> Value {
        Value::from_raw(unsafe { mlirOperationGetResult(self.operation, index as isize) })
    }

    pub fn region(&self, index: usize) -> RegionRef {
        unsafe { RegionRef::from_raw(mlirOperationGetRegion(self.operation, index as isize)) }
    }

    pub fn print(&self) -> StringRef {
        let mut string: Option<MlirStringRef> = None;

        unsafe extern "C" fn callback(string: MlirStringRef, data: *mut c_void) {
            *(data as *mut Option<MlirStringRef>) = Some(string);
        }

        unsafe {
            mlirOperationPrint(
                self.operation,
                Some(callback),
                &mut string as *mut _ as *mut c_void,
            );

            StringRef::from_raw(string.unwrap())
        }
    }

    pub fn dump(&self) {
        unsafe { mlirOperationDump(self.operation) }
    }

    pub(crate) unsafe fn from_raw(operation: MlirOperation) -> Self {
        Self { operation }
    }
}

impl Drop for Operation {
    fn drop(&mut self) {
        unsafe { mlirOperationDestroy(self.operation) };
    }
}

pub struct OperationRef<'o> {
    operation: ManuallyDrop<Operation>,
    _operation: PhantomData<&'o Operation>,
}

impl<'o> OperationRef<'o> {
    pub(crate) unsafe fn from_raw(operation: MlirOperation) -> Self {
        Self {
            operation: ManuallyDrop::new(Operation::from_raw(operation)),
            _operation: Default::default(),
        }
    }
}

impl<'o> Deref for OperationRef<'o> {
    type Target = Operation;

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
