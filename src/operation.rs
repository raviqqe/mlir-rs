use crate::{
    context::{Context, ContextRef},
    operation_state::OperationState,
};
use std::marker::PhantomData;

pub struct Operation<'c> {
    operation: mlir_sys::MlirOperation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Operation<'c> {
    pub fn new(state: OperationState) -> Self {
        Self {
            operation: unsafe { mlir_sys::mlirOperationCreate(&mut state.into_raw()) },
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
    use crate::location::Location;

    #[test]
    fn new() {
        Operation::new(OperationState::new(
            "foo",
            Location::new(&Context::new(), "foo", 42, 42),
        ));
    }
}
