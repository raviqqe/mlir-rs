use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{
    mlirArrayAttrGet, mlirArrayAttrGetElement, mlirArrayAttrGetNumElements, MlirAttribute,
};

/// An array attribute.
#[derive(Clone, Copy)]
pub struct ArrayAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> ArrayAttribute<'c> {
    /// Creates a dense i64 array attribute.
    pub fn new(context: &'c Context, values: &[Attribute<'c>]) -> Self {
        unsafe {
            Self::from_raw(mlirArrayAttrGet(
                context.to_raw(),
                values.len() as isize,
                values.as_ptr() as *const _ as *const _,
            ))
        }
    }

    /// Gets a length.
    pub fn len(&self) -> usize {
        (unsafe { mlirArrayAttrGetNumElements(self.attribute.to_raw()) }) as usize
    }

    /// Checks if an array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets an element.
    pub fn element(&self, index: usize) -> Result<Attribute<'c>, Error> {
        if index < self.len() {
            Ok(unsafe {
                Attribute::from_raw(mlirArrayAttrGetElement(
                    self.attribute.to_raw(),
                    index as isize,
                ))
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "array element",
                value: self.to_string(),
                index,
            })
        }
    }
}

attribute_traits!(ArrayAttribute, is_dense_i64_array, "dense i64 array");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element() {
        let context = Context::new();
        let attribute = ArrayAttribute::new(&context, &[1, 2, 3]);

        assert_eq!(attribute.element(0).unwrap(), 1);
        assert_eq!(attribute.element(1).unwrap(), 2);
        assert_eq!(attribute.element(2).unwrap(), 3);
        assert!(matches!(
            attribute.element(3),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn len() {
        let context = Context::new();
        let attribute = ArrayAttribute::new(&context, &[1, 2, 3]);

        assert_eq!(attribute.len(), 3);
    }
}
