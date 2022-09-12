use crate::{
    block::{BlockRef, BlockRefMut},
    context::{Context, ContextRef},
    location::Location,
    operation::{OperationRef, OperationRefMut},
    string_ref::StringRef,
};
use mlir_sys::{
    mlirModuleCreateEmpty, mlirModuleCreateParse, mlirModuleDestroy, mlirModuleGetBody,
    mlirModuleGetContext, mlirModuleGetOperation, MlirModule,
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

    pub fn parse(context: &Context, source: &str) -> Self {
        // TODO Should we allocate StringRef locally because source can be big?
        Self {
            module: unsafe {
                mlirModuleCreateParse(context.to_raw(), StringRef::from(source).to_raw())
            },
            _context: Default::default(),
        }
    }

    pub fn as_operation(&self) -> OperationRef {
        unsafe { OperationRef::from_raw(mlirModuleGetOperation(self.module)) }
    }

    pub fn as_operation_mut(&mut self) -> OperationRefMut {
        unsafe { OperationRefMut::from_raw(mlirModuleGetOperation(self.module)) }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirModuleGetContext(self.module)) }
    }

    pub fn body(&self) -> BlockRef {
        unsafe { BlockRef::from_raw(mlirModuleGetBody(self.module)) }
    }

    pub fn body_mut(&mut self) -> BlockRefMut {
        unsafe { BlockRefMut::from_raw(mlirModuleGetBody(self.module)) }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirModule {
        self.module
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
