use mlir_sys::{mlirDialectHandleInsertDialect, mlirGetDialectHandle__func__, MlirDialectHandle};

use crate::dialect_registry::DialectRegistry;

#[derive(Clone, Copy, Debug)]
pub struct DialectHandle {
    handle: MlirDialectHandle,
}

impl DialectHandle {
    pub fn func() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__func__()) }
    }

    pub fn insert_dialect(&self, registry: &DialectRegistry) {
        unsafe { mlirDialectHandleInsertDialect(self.handle, registry.to_raw()) }
    }

    pub(crate) unsafe fn from_raw(handle: MlirDialectHandle) -> Self {
        Self { handle }
    }
}
