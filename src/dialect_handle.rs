use crate::{
    context::Context, dialect::Dialect, dialect_registry::DialectRegistry, string_ref::StringRef,
};
use mlir_sys::{
    mlirDialectHandleGetNamespace, mlirDialectHandleInsertDialect, mlirDialectHandleLoadDialect,
    mlirDialectHandleRegisterDialect, mlirGetDialectHandle__async__, mlirGetDialectHandle__cf__,
    mlirGetDialectHandle__func__, mlirGetDialectHandle__gpu__, mlirGetDialectHandle__linalg__,
    mlirGetDialectHandle__llvm__, mlirGetDialectHandle__pdl__, mlirGetDialectHandle__quant__,
    mlirGetDialectHandle__scf__, mlirGetDialectHandle__shape__,
    mlirGetDialectHandle__sparse_tensor__, mlirGetDialectHandle__tensor__, MlirDialectHandle,
};

/// A dialect handle.
#[derive(Clone, Copy, Debug)]
pub struct DialectHandle {
    raw: MlirDialectHandle,
}

impl DialectHandle {
    pub fn r#async() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__async__()) }
    }

    pub fn cf() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__cf__()) }
    }

    pub fn func() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__func__()) }
    }

    pub fn gpu() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__gpu__()) }
    }

    pub fn linalg() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__linalg__()) }
    }

    pub fn llvm() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__llvm__()) }
    }

    pub fn pdl() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__pdl__()) }
    }

    pub fn quant() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__quant__()) }
    }

    pub fn scf() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__scf__()) }
    }

    pub fn shape() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__shape__()) }
    }

    pub fn sparse_tensor() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__sparse_tensor__()) }
    }

    pub fn tensor() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__tensor__()) }
    }

    pub fn namespace(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirDialectHandleGetNamespace(self.raw)) }
    }

    pub fn insert_dialect(&self, registry: &DialectRegistry) {
        unsafe { mlirDialectHandleInsertDialect(self.raw, registry.to_raw()) }
    }

    pub fn load_dialect<'c>(&self, context: &'c Context) -> Dialect<'c> {
        unsafe { Dialect::from_raw(mlirDialectHandleLoadDialect(self.raw, context.to_raw())) }
    }

    pub fn register_dialect(&self, context: &Context) {
        unsafe { mlirDialectHandleRegisterDialect(self.raw, context.to_raw()) }
    }

    pub(crate) unsafe fn from_raw(handle: MlirDialectHandle) -> Self {
        Self { raw: handle }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn func() {
        DialectHandle::func();
    }

    #[test]
    fn llvm() {
        DialectHandle::llvm();
    }

    #[test]
    fn namespace() {
        DialectHandle::func().namespace();
    }

    #[test]
    fn insert_dialect() {
        let registry = DialectRegistry::new();

        DialectHandle::func().insert_dialect(&registry);
    }

    #[test]
    fn load_dialect() {
        let context = Context::new();

        DialectHandle::func().load_dialect(&context);
    }

    #[test]
    fn register_dialect() {
        let context = Context::new();

        DialectHandle::func().register_dialect(&context);
    }
}
