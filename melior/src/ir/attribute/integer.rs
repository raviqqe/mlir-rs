use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Error,
};
use mlir_sys::{
    mlirIntegerAttrGet, mlirIntegerAttrGetValueInt, mlirIntegerAttrGetValueSInt,
    mlirIntegerAttrGetValueUInt, MlirAttribute,
};

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

    /// Returns a value.
    pub fn value(&self) -> i64 {
        unsafe { mlirIntegerAttrGetValueInt(self.to_raw()) }
    }

    /// Returns a value as a signed integer.
    pub fn signed_value(&self) -> i64 {
        unsafe { mlirIntegerAttrGetValueSInt(self.to_raw()) }
    }

    /// Returns a value as an unsigned integer.
    pub fn unsigned_value(&self) -> u64 {
        unsafe { mlirIntegerAttrGetValueUInt(self.to_raw()) }
    }
}

attribute_traits!(IntegerAttribute, is_integer, "integer");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::r#type::IntegerType, test::create_test_context};

    #[test]
    fn value() {
        let context = create_test_context();

        assert_eq!(
            IntegerAttribute::new(42, IntegerType::new(&context, 64).into()).value(),
            42
        );
    }
}
