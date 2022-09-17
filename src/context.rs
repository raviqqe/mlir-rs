use crate::{
    dialect::{self, Dialect},
    string_ref::StringRef,
};
use mlir_sys::{
    mlirContextAppendDialectRegistry, mlirContextCreate, mlirContextDestroy,
    mlirContextGetNumRegisteredDialects, mlirContextGetOrLoadDialect,
    mlirContextLoadAllAvailableDialects, MlirContext,
};
use std::{marker::PhantomData, ops::Deref};

/// A context of IR, dialects, and passes.
///
/// Contexts own various objects, such as types, locations, and dialect
/// instances.
#[derive(Debug)]
pub struct Context {
    r#ref: ContextRef<'static>,
}

impl Context {
    /// Creates a context.
    pub fn new() -> Self {
        Self {
            r#ref: unsafe { ContextRef::from_raw(mlirContextCreate()) },
        }
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

impl Deref for Context {
    type Target = ContextRef<'static>;

    fn deref(&self) -> &Self::Target {
        &self.r#ref
    }
}

/// A reference to a context.
#[derive(Clone, Copy, Debug)]
pub struct ContextRef<'a> {
    raw: MlirContext,
    _reference: PhantomData<&'a Context>,
}

impl<'a> ContextRef<'a> {
    /// Gets a number of registered dialects.
    pub fn registered_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumRegisteredDialects(self.raw) as usize }
    }

    /// Gets or loads a dialect.
    pub fn get_or_load_dialect(&self, name: &str) -> Dialect {
        unsafe {
            Dialect::from_raw(mlirContextGetOrLoadDialect(
                self.raw,
                StringRef::from(name).to_raw(),
            ))
        }
    }

    /// Appends a dialect registry.
    pub fn append_dialect_registry(&self, registry: &dialect::Registry) {
        unsafe { mlirContextAppendDialectRegistry(self.raw, registry.to_raw()) }
    }

    /// Loads all available dialects.
    pub fn load_all_available_dialects(&self) {
        unsafe { mlirContextLoadAllAvailableDialects(self.raw) }
    }

    pub(crate) unsafe fn to_raw(self) -> MlirContext {
        self.raw
    }

    pub(crate) unsafe fn from_raw(raw: MlirContext) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
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

        context.append_dialect_registry(&dialect::Registry::new());
    }
}
