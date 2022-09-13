use crate::{dialect::Dialect, dialect_registry::DialectRegistry, string_ref::StringRef};
use mlir_sys::{
    mlirContextAppendDialectRegistry, mlirContextCreate, mlirContextDestroy,
    mlirContextGetNumRegisteredDialects, mlirContextGetOrLoadDialect,
    mlirContextLoadAllAvailableDialects, mlirRegisterAllLLVMTranslations, MlirContext,
};
use std::{marker::PhantomData, mem::ManuallyDrop, ops::Deref};

pub struct Context {
    raw: MlirContext,
}

impl Context {
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirContextCreate() },
        }
    }

    pub fn registered_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumRegisteredDialects(self.raw) as usize }
    }

    pub fn get_or_load_dialect(&self, name: &str) -> Dialect {
        unsafe {
            Dialect::from_raw(mlirContextGetOrLoadDialect(
                self.raw,
                StringRef::from(name).to_raw(),
            ))
        }
    }

    pub fn register_all_llvm_translations(&self) {
        unsafe { mlirRegisterAllLLVMTranslations(self.raw) }
    }

    pub fn append_dialect_registry(&self, registry: &DialectRegistry) {
        unsafe { mlirContextAppendDialectRegistry(self.raw, registry.to_raw()) }
    }

    pub fn load_all_available_dialects(&self) {
        unsafe { mlirContextLoadAllAvailableDialects(self.raw) }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirContext {
        self.raw
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { mlirContextDestroy(self.raw) };
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ContextRef<'c> {
    raw: ManuallyDrop<Context>,
    _reference: PhantomData<&'c Context>,
}

impl<'c> ContextRef<'c> {
    pub(crate) unsafe fn from_raw(context: MlirContext) -> Self {
        Self {
            raw: ManuallyDrop::new(Context { raw: context }),
            _reference: Default::default(),
        }
    }
}

impl<'c> Deref for ContextRef<'c> {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.raw
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
