use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Error,
};
use mlir_sys::{
    mlirDenseElementsAttrGet, mlirDenseElementsAttrGetInt64Value, mlirElementsAttrGetNumElements,
    MlirAttribute,
};

/// A dense elements attribute.
#[derive(Clone, Copy)]
pub struct DenseElementsAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> DenseElementsAttribute<'c> {
    /// Creates a dense elements attribute.
    pub fn new(r#type: Type<'c>, values: &[Attribute<'c>]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrGet(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr() as *const _ as *const _,
            ))
        }
    }

    /// Gets a length.
    pub fn len(&self) -> usize {
        (unsafe { mlirElementsAttrGetNumElements(self.attribute.to_raw()) }) as usize
    }

    /// Checks if an array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets an i64 element.
    pub fn i64_element(&self, index: usize) -> Result<i64, Error> {
        if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetInt64Value(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }
}

attribute_traits!(DenseElementsAttribute, is_dense_elements, "dense elements");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            attribute::IntegerAttribute,
            r#type::{IntegerType, MemRefType},
        },
        Context,
    };

    #[test]
    fn i64_element() {
        let context = Context::new();
        let attribute = DenseElementsAttribute::new(
            MemRefType::new(Type::index(&context), &[3], None, None).into(),
            &[IntegerAttribute::new(42, IntegerType::new(&context, 64).into()).into()],
        );

        assert_eq!(attribute.i64_element(0), Ok(42));
        assert_eq!(attribute.i64_element(1), Ok(42));
        assert_eq!(attribute.i64_element(2), Ok(42));
        assert_eq!(
            attribute.i64_element(3),
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: attribute.to_string(),
                index: 3,
            })
        );
    }

    #[test]
    fn len() {
        let context = Context::new();
        let attribute = DenseElementsAttribute::new(
            MemRefType::new(Type::index(&context), &[3], None, None).into(),
            &[IntegerAttribute::new(0, IntegerType::new(&context, 64).into()).into()],
        );

        assert_eq!(attribute.len(), 3);
    }
}
