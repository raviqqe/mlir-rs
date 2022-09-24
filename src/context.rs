use crate::{
    dialect::{self, Dialect},
    string_ref::StringRef,
};
use mlir_sys::{
    mlirContextAppendDialectRegistry, mlirContextCreate, mlirContextDestroy,
    mlirContextEnableMultithreading, mlirContextEqual, mlirContextGetAllowUnregisteredDialects,
    mlirContextGetNumLoadedDialects, mlirContextGetNumRegisteredDialects,
    mlirContextGetOrLoadDialect, mlirContextIsRegisteredOperation,
    mlirContextLoadAllAvailableDialects, mlirContextSetAllowUnregisteredDialects, MlirContext,
};
use std::{marker::PhantomData, mem::transmute, ops::Deref};

/// A context of IR, dialects, and passes.
///
/// Contexts own various objects, such as types, locations, and dialect
/// instances.
#[derive(Debug)]
pub struct Context {
    raw: MlirContext,
}

impl Context {
    /// Creates a context.
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirContextCreate() },
        }
    }

    /// Gets a number of registered dialects.
    pub fn registered_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumRegisteredDialects(self.raw) as usize }
    }

    /// Gets a number of loaded dialects.
    pub fn loaded_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumLoadedDialects(self.raw) as usize }
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

    /// Enables multi-threading.
    pub fn enable_multi_threading(&self, enabled: bool) {
        unsafe { mlirContextEnableMultithreading(self.raw, enabled) }
    }

    /// Returns `true` if unregistered dialects are allowed.
    pub fn allow_unregistered_dialects(&self) -> bool {
        unsafe { mlirContextGetAllowUnregisteredDialects(self.raw) }
    }

    /// Set if unregistered dialects are allowed.
    pub fn set_allow_unregistered_dialects(&self, allowed: bool) {
        unsafe { mlirContextSetAllowUnregisteredDialects(self.raw, allowed) }
    }

    /// Returns `true` if a given operation is registered in a context.
    pub fn is_registered_operation(&self, name: &str) -> bool {
        unsafe { mlirContextIsRegisteredOperation(self.raw, StringRef::from(name).to_raw()) }
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

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirContextEqual(self.raw, other.raw) }
    }
}

impl Eq for Context {}

/// A reference to a context.
#[derive(Clone, Copy, Debug)]
pub struct ContextRef<'a> {
    raw: MlirContext,
    _reference: PhantomData<&'a Context>,
}

impl<'a> ContextRef<'a> {
    pub(crate) unsafe fn from_raw(raw: MlirContext) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
    }
}

impl<'a> Deref for ContextRef<'a> {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}

impl<'a> PartialEq for ContextRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirContextEqual(self.raw, other.raw) }
    }
}

impl<'a> Eq for ContextRef<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Context::new();
    }

    #[test]
    fn registered_dialect_count() {
        let context = Context::new();

        assert_eq!(context.registered_dialect_count(), 1);
    }

    #[test]
    fn loaded_dialect_count() {
        let context = Context::new();

        assert_eq!(context.loaded_dialect_count(), 1);
    }

    #[test]
    fn append_dialect_registry() {
        let context = Context::new();

        context.append_dialect_registry(&dialect::Registry::new());
    }

    #[test]
    fn is_registered_operation() {
        let context = Context::new();

        assert!(context.is_registered_operation("builtin.module"));
    }

    #[test]
    fn is_not_registered_operation() {
        let context = Context::new();

        assert!(!context.is_registered_operation("func.func"));
    }

    #[test]
    fn enable_multi_threading() {
        let context = Context::new();

        context.enable_multi_threading(true);
    }

    #[test]
    fn disable_multi_threading() {
        let context = Context::new();

        context.enable_multi_threading(false);
    }

    #[test]
    fn allow_unregistered_dialects() {
        let context = Context::new();

        assert!(!context.allow_unregistered_dialects());
    }

    #[test]
    fn set_allow_unregistered_dialects() {
        let context = Context::new();

        context.set_allow_unregistered_dialects(true);

        assert!(context.allow_unregistered_dialects());
    }
}
