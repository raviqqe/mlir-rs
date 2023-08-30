//! Dialect handles, instances, and registry.

pub mod arith;
pub mod cf;
pub mod func;
mod handle;
pub mod index;
pub mod llvm;
pub mod memref;
mod registry;
pub mod scf;

pub use self::{handle::DialectHandle, registry::DialectRegistry};
use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{mlirDialectEqual, mlirDialectGetContext, mlirDialectGetNamespace, MlirDialect};
use std::marker::PhantomData;

#[cfg(feature = "ods-dialects")]
pub mod ods;

/// A dialect.
#[derive(Clone, Copy, Debug)]
pub struct Dialect<'c> {
    raw: MlirDialect,
    _context: PhantomData<&'c Context>,
}

impl<'c> Dialect<'c> {
    /// Gets a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirDialectGetContext(self.raw)) }
    }

    /// Gets a namespace.
    // TODO Return &str.
    pub fn namespace(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirDialectGetNamespace(self.raw)) }
    }

    /// Creates a dialect from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(dialect: MlirDialect) -> Self {
        Self {
            raw: dialect,
            _context: Default::default(),
        }
    }
}

impl<'c> PartialEq for Dialect<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirDialectEqual(self.raw, other.raw) }
    }
}

impl<'c> Eq for Dialect<'c> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal() {
        let context = Context::new();

        assert_eq!(
            DialectHandle::func().load_dialect(&context),
            DialectHandle::func().load_dialect(&context)
        );
    }

    #[test]
    fn not_equal() {
        let context = Context::new();

        assert_ne!(
            DialectHandle::func().load_dialect(&context),
            DialectHandle::llvm().load_dialect(&context)
        );
    }
}
