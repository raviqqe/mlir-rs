use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Context, Error,
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
        unsafe { Self::from_raw(mlirIntegerAttrGet(r#type.to_raw(), integer)) }
    }

    pub(crate) unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
}

impl<'c> AttributeLike<'c> for Integer<'c> {
    unsafe fn to_raw(&self) -> MlirAttribute {
        self.raw
    }
}

impl<'c> TryFrom<Attribute<'c>> for Integer<'c> {
    fn from(attribute: Attribute<'c>) -> Result<Self, Error> {
        if attribute.is_integer() {
            Ok(Self::from_raw(attribute.to_raw()))
        } else {
            Err(Error::IntegerAttributeExpected())
        }
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
