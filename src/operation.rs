use crate::{
    context::{Context, ContextRef},
    operation_state::OperationState,
    region::RegionRef,
    string_ref::StringRef,
    value::Value,
};
use mlir_sys::{
    mlirOperationCreate, mlirOperationDestroy, mlirOperationDump, mlirOperationGetContext,
    mlirOperationGetRegion, mlirOperationGetResult, mlirOperationPrint, MlirOperation,
    MlirStringRef,
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
        Self {
            operation,
            _context: Default::default(),
        }
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
