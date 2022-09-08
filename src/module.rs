use crate::{
    context::{Context, ContextRef},
    location::Location,
    operation::OperationRef,
};
use mlir_sys::{
    mlirModuleCreateEmpty, mlirModuleDestroy, mlirModuleGetContext, mlirModuleGetOperation,
    MlirModule,
};
use std::marker::PhantomData;

pub struct Module<'c> {
    module: MlirModule,
    _context: PhantomData<&'c Context>,
}

impl<'c> Module<'c> {
    pub fn new(location: Location) -> Self {
        Self {
            module: unsafe { mlirModuleCreateEmpty(location.to_raw()) },
            _context: Default::default(),
        }
    }
    pub fn as_operation<'a>(&'a self) -> OperationRef<'c, 'a> {
        unsafe { OperationRef::from_raw(mlirModuleGetOperation(self.module)) }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirModuleGetContext(self.module)) }
    }
}

impl<'c> Drop for Module<'c> {
    fn drop(&mut self) {
        unsafe { mlirModuleDestroy(self.module) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42));
    }

    #[test]
    fn context() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42)).context();
    }
}
