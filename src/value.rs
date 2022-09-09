use crate::context::Context;
use mlir_sys::MlirValue;
use std::marker::PhantomData;

pub struct Value<'c> {
    value: MlirValue,
    _context: PhantomData<&'c Context>,
}

impl<'c> Value<'c> {
    pub(crate) fn from_raw(value: MlirValue) -> Self {
        Self {
            value,
            _context: Default::default(),
        }
    }

    pub(crate) fn to_raw(&self) -> MlirValue {
        self.value
    }
}
