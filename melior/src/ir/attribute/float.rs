use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Context, Error,
};
use mlir_sys::{mlirFloatAttrDoubleGet, MlirAttribute};
use std::{
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

/// An float attribute.
// Attributes are always values but their internal storage is owned by contexts.
#[derive(Clone, Copy)]
pub struct Float<'c> {
    raw: MlirAttribute,
    _context: PhantomData<&'c Context>,
}

impl<'c> Float<'c> {
    /// Creates an float.
    pub fn new(context: &'c Context, number: f64, r#type: Type<'c>) -> Self {
        unsafe {
            Self::from_raw(mlirFloatAttrDoubleGet(
                context.to_raw(),
                r#type.to_raw(),
                number,
            ))
        }
    }

    unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
}

impl<'c> AttributeLike<'c> for Float<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.raw
    }
}

impl<'c> TryFrom<Attribute<'c>> for Float<'c> {
    type Error = Error;

    fn try_from(attribute: Attribute<'c>) -> Result<Self, Self::Error> {
        if attribute.is_float() {
            Ok(unsafe { Self::from_raw(attribute.to_raw()) })
        } else {
            Err(Error::AttributeExpected("float", format!("{}", attribute)))
        }
    }
}

impl<'c> Display for Float<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(&Attribute::from(*self), formatter)
    }
}

impl<'c> Debug for Float<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}
