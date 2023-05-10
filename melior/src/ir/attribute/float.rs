use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Context, Error,
};
use mlir_sys::{mlirFloatAttrDoubleGet, MlirAttribute};

/// A float attribute.
#[derive(Clone, Copy)]
pub struct FloatAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> FloatAttribute<'c> {
    /// Creates a float attribute.
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
            attribute: Attribute::from_raw(raw),
        }
    }
}

attribute_traits!(FloatAttribute, is_float, "float");
