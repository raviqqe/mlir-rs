use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{mlirDialectEqual, mlirDialectGetContext, mlirDialectGetNamespace, MlirDialect};
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug)]
pub struct Dialect<'c> {
    dialect: MlirDialect,
    _context: PhantomData<&'c Context>,
}

impl<'c> Dialect<'c> {
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirDialectGetContext(self.dialect)) }
    }

    pub fn namespace(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirDialectGetNamespace(self.dialect)) }
    }

    pub(crate) unsafe fn from_raw(dialect: MlirDialect) -> Self {
        Self {
            dialect,
            _context: Default::default(),
        }
    }
}

impl<'c> PartialEq for Dialect<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirDialectEqual(self.dialect, other.dialect) }
    }
}

impl<'c> Eq for Dialect<'c> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect_handle::DialectHandle;

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
