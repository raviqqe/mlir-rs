use crate::{
    block::BlockRef,
    context::{Context, ContextRef},
    location::Location,
    operation::OperationRef,
    string_ref::StringRef,
};
use mlir_sys::{
    mlirModuleCreateEmpty, mlirModuleCreateParse, mlirModuleDestroy, mlirModuleGetBody,
    mlirModuleGetContext, mlirModuleGetOperation, MlirModule,
};
use std::marker::PhantomData;

pub struct Module<'c> {
    raw: MlirModule,
    _context: PhantomData<&'c Context>,
}

impl<'c> Module<'c> {
    pub fn new(location: Location) -> Self {
        Self {
            raw: unsafe { mlirModuleCreateEmpty(location.to_raw()) },
            _context: Default::default(),
        }
    }

    pub fn parse(context: &Context, source: &str) -> Self {
        // TODO Should we allocate StringRef locally because sources can be big?
        Self {
            raw: unsafe {
                mlirModuleCreateParse(context.to_raw(), StringRef::from(source).to_raw())
            },
            _context: Default::default(),
        }
    }

    pub fn as_operation(&self) -> OperationRef {
        unsafe { OperationRef::from_raw(mlirModuleGetOperation(self.raw)) }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirModuleGetContext(self.raw)) }
    }

    pub fn body(&self) -> BlockRef {
        unsafe { BlockRef::from_raw(mlirModuleGetBody(self.raw)) }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirModule {
        self.raw
    }
}

impl<'c> Drop for Module<'c> {
    fn drop(&mut self) {
        unsafe { mlirModuleDestroy(self.raw) };
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
