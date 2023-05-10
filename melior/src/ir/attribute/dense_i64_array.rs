use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{mlirDenseI64ArrayGet, MlirAttribute};
use std::fmt::{self, Debug, Display, Formatter};

/// An dense i64 array attribute.
// Attributes are always values but their internal storage is owned by contexts.
#[derive(Clone, Copy)]
pub struct DenseI64ArrayAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> DenseI64ArrayAttribute<'c> {
    /// Creates an dense i64 array.
    pub fn new(context: &'c Context, values: &[i64]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseI64ArrayGet(
                context.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            attribute: Attribute::from_raw(raw),
        }
    }
}

impl<'c> AttributeLike<'c> for DenseI64ArrayAttribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.attribute.to_raw()
    }
}

impl<'c> TryFrom<Attribute<'c>> for DenseI64ArrayAttribute<'c> {
    type Error = Error;

    fn try_from(attribute: Attribute<'c>) -> Result<Self, Self::Error> {
        if attribute.is_dense_i64_array() {
            Ok(unsafe { Self::from_raw(attribute.to_raw()) })
        } else {
            Err(Error::AttributeExpected(
                "dense i64 array",
                format!("{}", attribute),
            ))
        }
    }
}

impl<'c> Display for DenseI64ArrayAttribute<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(&self.attribute, formatter)
    }
}

impl<'c> Debug for DenseI64ArrayAttribute<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}
