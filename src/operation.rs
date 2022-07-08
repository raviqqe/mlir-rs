use crate::{
    context::{Context, ContextRef},
    operation_state::OperationState,
    value::Value,
};
use mlir_sys::{
    mlirOperationCreate, mlirOperationDestroy, mlirOperationGetContext, mlirOperationGetResult,
    MlirOperation,
};
use std::marker::PhantomData;

pub struct Operation<'c> {
    operation: MlirOperation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Operation<'c> {
    pub fn new(state: OperationState) -> Self {
        let mut state = state.to_raw();

        Self {
            operation: unsafe { mlirOperationCreate(&mut state) },
            _context: Default::default(),
        }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirOperationGetContext(self.operation)) }
    }

    pub fn result(&self, index: usize) -> Value {
        Value::from_raw(unsafe { mlirOperationGetResult(self.operation, index as isize) })
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
            Location::new(&Context::new(), "foo", 42, 42),
        ));
    }
}
