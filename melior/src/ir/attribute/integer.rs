use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Error,
};
use mlir_sys::{mlirIntegerAttrGet, MlirAttribute};
use std::fmt::{self, Debug, Display, Formatter};

/// An integer attribute.
#[derive(Clone, Copy)]
pub struct IntegerAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> IntegerAttribute<'c> {
    /// Creates an integer attribute.
    pub fn new(integer: i64, r#type: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirIntegerAttrGet(r#type.to_raw(), integer)) }
    }

    unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            attribute: Attribute::from_raw(raw),
        }
    }
}

impl<'c> AttributeLike<'c> for IntegerAttribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.attribute.to_raw()
    }
}

impl<'c> TryFrom<Attribute<'c>> for IntegerAttribute<'c> {
    type Error = Error;

    fn try_from(attribute: Attribute<'c>) -> Result<Self, Self::Error> {
        if attribute.is_integer() {
            Ok(unsafe { Self::from_raw(attribute.to_raw()) })
        } else {
            Err(Error::AttributeExpected(
                "integer",
                format!("{}", attribute),
            ))
        }
    }
}

impl<'c> Display for IntegerAttribute<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(&self.attribute, formatter)
    }
}

impl<'c> Debug for IntegerAttribute<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}
