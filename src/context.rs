use crate::dialect_registry::DialectRegistry;
use mlir_sys::{
    mlirContextAppendDialectRegistry, mlirContextCreate, mlirContextDestroy, MlirContext,
};
use std::{marker::PhantomData, mem::ManuallyDrop, ops::Deref};

pub struct Context {
    context: MlirContext,
}

impl Context {
    pub fn new() -> Self {
        Self {
            context: unsafe { mlirContextCreate() },
        }
    }

    pub fn append_dialect_registry(&self, registry: &DialectRegistry) {
        unsafe { mlirContextAppendDialectRegistry(self.to_raw(), registry.to_raw()) }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirContext {
        self.context
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { mlirContextDestroy(self.context) };
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ContextRef<'c> {
    context: ManuallyDrop<Context>,
    _context: PhantomData<&'c Context>,
}

impl<'c> ContextRef<'c> {
    pub(crate) unsafe fn from_raw(context: MlirContext) -> Self {
        Self {
            context: ManuallyDrop::new(Context { context }),
            _context: Default::default(),
        }
    }
}

impl<'c> Deref for ContextRef<'c> {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Context::new();
    }

    #[test]
    fn append_dialect_registry() {
        let context = Context::new();

        context.append_dialect_registry(&DialectRegistry::new());
    }
}
