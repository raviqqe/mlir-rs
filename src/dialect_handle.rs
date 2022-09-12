use mlir_sys::{mlirGetDialectHandle__func__, MlirDialect, MlirDialectHandle};

#[derive(Clone, Copy, Debug)]
pub struct DialectHandle {
    handle: MlirDialectHandle,
}

impl DialectHandle {
    pub fn func(&self) -> Self {
        Self::from_raw(unsafe { mlirGetDialectHandle__func__() })
    }

    pub(crate) unsafe fn from_raw(dialect: MlirDialect) -> Self {
        Self {
            dialect,
            _context: Default::default(),
        }
    }
}
