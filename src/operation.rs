use crate::{
    context::{Context, ContextRef},
    operation_state::OperationState,
    string_ref::StringRef,
    value::Value,
};
use mlir_sys::{
    mlirOperationCreate, mlirOperationDestroy, mlirOperationGetContext, mlirOperationGetResult,
    mlirOperationPrint, MlirOperation, MlirStringRef,
};
use std::{ffi::c_void, marker::PhantomData, mem::ManuallyDrop, ops::Deref};

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

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirOperationGetContext(self.operation)) }
    }

    pub fn result(&self, index: usize) -> Value {
        Value::from_raw(unsafe { mlirOperationGetResult(self.operation, index as isize) })
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

    pub(crate) unsafe fn from_raw(operation: MlirOperation) -> Self {
        Self {
            operation,
            _context: Default::default(),
        }
    }
}

pub struct OperationRef<'c, 'o> {
    operation: ManuallyDrop<Operation<'c>>,
    _operation: PhantomData<&'o Operation<'c>>,
}

impl<'c, 'o> OperationRef<'c, 'o> {
    pub(crate) unsafe fn from_raw(operation: MlirOperation) -> Self {
        Self {
            operation: ManuallyDrop::new(Operation::from_raw(operation)),
            _operation: Default::default(),
        }
    }
}

impl<'c, 'o> Deref for OperationRef<'c, 'o> {
    type Target = Operation<'c>;

    fn deref(&self) -> &Self::Target {
        &self.operation
    }
}

impl<'c> Drop for Operation<'c> {
    fn drop(&mut self) {
        unsafe { mlirOperationDestroy(self.operation) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::location::Location;

    #[test]
    fn new() {
        Operation::new(OperationState::new(
            "foo",
            Location::unknown(&Context::new()),
        ));
    }
}
