use crate::{
    context::{Context, ContextRef},
    location::Location,
};
use std::{marker::PhantomData, mem::forget};

pub struct Operation<'c> {
    operation: mlir_sys::MlirOperation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Operation<'c> {
    pub fn new(state: MlirOperationState) -> Self {
        let state = &mut state;

        forget(MlirOperationState);

        Self {
            operation: unsafe { mlir_sys::mlirOperationCreate(state.as_raw_mut()) },
            _context: Default::default(),
        }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlir_sys::mlirOperationGetContext(self.operation)) }
    }
}

impl<'c> Drop for Operation<'c> {
    fn drop(&mut self) {
        unsafe { mlir_sys::mlirOperationDestroy(self.operation) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Operation::new(Location::new(&Context::new(), "foo", 42, 42));
    }

    #[test]
    fn context() {
        Operation::new(Location::new(&Context::new(), "foo", 42, 42)).context();
    }
}
