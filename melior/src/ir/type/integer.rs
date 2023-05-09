use super::TypeLike;
use crate::{ir::Type, Context, Error};
use mlir_sys::{mlirIntegerTypeGet, mlirIntegerTypeGetWidth, MlirType};
use std::fmt::{self, Display, Formatter};

/// A integer type.
#[derive(Clone, Copy, Debug)]
pub struct Integer<'c> {
    r#type: Type<'c>,
}

impl<'c> Integer<'c> {
    /// Creates a integer type.
    pub fn new(context: &'c Context, bits: u32) -> Self {
        Self {
            r#type: unsafe { Type::from_raw(mlirIntegerTypeGet(context.to_raw(), bits)) },
        }
    }

    /// Gets a bit width.
    pub fn width(&self) -> u32 {
        unsafe { mlirIntegerTypeGetWidth(self.to_raw()) }
    }
}

impl<'c> TypeLike<'c> for Integer<'c> {
    fn to_raw(&self) -> MlirType {
        self.r#type.to_raw()
    }
}

impl<'c> Display for Integer<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Type::from(*self).fmt(formatter)
    }
}

impl<'c> TryFrom<Type<'c>> for Integer<'c> {
    type Error = Error;

    fn try_from(r#type: Type<'c>) -> Result<Self, Self::Error> {
        if r#type.is_integer() {
            Ok(Self { r#type })
        } else {
            Err(Error::TypeExpected("integer", r#type.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_width() {
        let context = Context::new();

        assert_eq!(Integer::new(&context, 64).width(), 64);
    }
}
