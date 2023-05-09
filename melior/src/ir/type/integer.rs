use super::TypeLike;
use crate::{ir::Type, Context, Error};
use mlir_sys::{
    mlirIndexTypeGet, mlirIntegerTypeGet, mlirIntegerTypeGetWidth, mlirIntegerTypeIsSigned,
    mlirIntegerTypeIsSignless, mlirIntegerTypeIsUnsigned, mlirIntegerTypeSignedGet,
    mlirIntegerTypeUnsignedGet, MlirType,
};
use std::fmt::{self, Display, Formatter};

/// A integer type.
#[derive(Clone, Copy, Debug)]
pub struct Integer<'c> {
    r#type: Type<'c>,
}

impl<'c> Integer<'c> {
    /// Creates an integer type.
    pub fn new(context: &'c Context, bits: u32) -> Self {
        Self {
            r#type: unsafe { Type::from_raw(mlirIntegerTypeGet(context.to_raw(), bits)) },
        }
    }

    /// Creates a signed integer type.
    pub fn signed(context: &'c Context, bits: u32) -> Self {
        unsafe { Self::from_raw(mlirIntegerTypeSignedGet(context.to_raw(), bits)) }
    }

    /// Creates an unsigned integer type.
    pub fn unsigned(context: &'c Context, bits: u32) -> Self {
        unsafe { Self::from_raw(mlirIntegerTypeUnsignedGet(context.to_raw(), bits)) }
    }

    /// Gets a bit width.
    pub fn width(&self) -> u32 {
        unsafe { mlirIntegerTypeGetWidth(self.to_raw()) }
    }

    /// Checks if an integer type is signed.
    pub fn is_signed(&self) -> bool {
        unsafe { mlirIntegerTypeIsSigned(self.to_raw()) }
    }

    /// Checks if an integer type is signless.
    pub fn is_signless(&self) -> bool {
        unsafe { mlirIntegerTypeIsSignless(self.to_raw()) }
    }

    /// Checks if an integer type is unsigned.
    pub fn is_unsigned(&self) -> bool {
        unsafe { mlirIntegerTypeIsUnsigned(self.to_raw()) }
    }

    fn from_raw(raw: MlirType) -> Self {
        Self {
            r#type: unsafe { Type::from_raw(raw) },
        }
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
    fn new() {
        assert!(Integer::new(&Context::new(), 64).is_integer());
    }

    #[test]
    fn signed() {
        assert!(Integer::signed(&Context::new(), 64).is_integer());
    }

    #[test]
    fn unsigned() {
        assert!(Integer::unsigned(&Context::new(), 64).is_integer());
    }

    #[test]
    fn signed_integer() {
        let context = Context::new();

        assert_eq!(
            Type::from(Integer::signed(&context, 42)),
            Type::parse(&context, "si42").unwrap()
        );
    }

    #[test]
    fn unsigned_integer() {
        let context = Context::new();

        assert_eq!(
            Type::from(Integer::unsigned(&context, 42)),
            Type::parse(&context, "ui42").unwrap()
        );
    }

    #[test]
    fn get_width() {
        let context = Context::new();

        assert_eq!(Integer::new(&context, 64).width(), 64);
    }

    #[test]
    fn check_sign() {
        let context = Context::new();
        let signless = Integer::new(&context, 42);
        let signed = Integer::signed(&context, 42);
        let unsigned = Integer::unsigned(&context, 42);

        assert!(signless.is_signless());
        assert!(!signed.is_signless());
        assert!(!unsigned.is_signless());

        assert!(!signless.is_signed());
        assert!(signed.is_signed());
        assert!(!unsigned.is_signed());

        assert!(!signless.is_unsigned());
        assert!(!signed.is_unsigned());
        assert!(unsigned.is_unsigned());
    }
}
