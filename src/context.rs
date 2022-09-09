use crate::{dialect::Dialect, dialect_registry::DialectRegistry, string_ref::StringRef};
use mlir_sys::{
    mlirContextAppendDialectRegistry, mlirContextCreate, mlirContextDestroy,
    mlirContextGetOrLoadDialect, mlirContextLoadAllAvailableDialects,
    mlirRegisterAllLLVMTranslations, MlirContext,
};
use std::{marker::PhantomData, mem::ManuallyDrop, ops::Deref};

pub struct Context {
    context: MlirContext,
}

impl Context {
    pub fn new() -> Self {
        let context = unsafe { mlirContextCreate() };

        unsafe { mlirRegisterAllLLVMTranslations(context) }

        Self { context }
    }

    pub fn get_or_load_dialect(&self, name: &str) -> Dialect {
        unsafe {
            Dialect::from_raw(mlirContextGetOrLoadDialect(
                self.context,
                StringRef::from(name).to_raw(),
            ))
        }
    }

    pub fn append_dialect_registry(&self, registry: &DialectRegistry) {
        unsafe { mlirContextAppendDialectRegistry(self.context, registry.to_raw()) }
    }

    pub fn load_all_available_dialects(&self) {
        unsafe { mlirContextLoadAllAvailableDialects(self.context) }
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
