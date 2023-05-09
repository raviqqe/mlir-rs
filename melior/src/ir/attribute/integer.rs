use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Context,
};
use mlir_sys::{mlirIntegerAttrGet, MlirAttribute};
use std::{
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

/// An integer attribute.
// Attributes are always values but their internal storage is owned by contexts.
#[derive(Clone, Copy)]
pub struct Integer<'c> {
    raw: MlirAttribute,
    _context: PhantomData<&'c Context>,
}

impl<'c> Integer<'c> {
    /// Creates an integer.
    pub fn new(integer: i64, r#type: Type<'c>) -> Self {
        Self {
            raw: unsafe { mlirIntegerAttrGet(r#type.to_raw(), integer) },
            _context: Default::default(),
        }
    }
}

impl<'c> AttributeLike<'c> for Integer<'c> {
    unsafe fn to_raw(&self) -> MlirAttribute {
        self.raw
    }
}

impl<'c> Display for Integer<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(&Attribute::from(*self), formatter)
    }
}

impl<'c> Debug for Integer<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}
