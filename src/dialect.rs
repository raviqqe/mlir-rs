use crate::context::{Context, ContextRef};
use mlir_sys::{mlirDialectGetContext, MlirDialect};
use std::marker::PhantomData;

pub struct Dialect<'c> {
    dialect: MlirDialect,
    _context: PhantomData<&'c Context>,
}

impl<'c> Dialect<'c> {
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirDialectGetContext(self.dialect)) }
    }

    pub(crate) unsafe fn from_raw(dialect: MlirDialect) -> Self {
        Self {
            dialect,
            _context: Default::default(),
        }
    }
}
