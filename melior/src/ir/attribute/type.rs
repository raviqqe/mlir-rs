use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Error,
};
use mlir_sys::{mlirTypeAttrGet, MlirAttribute};

/// A type attribute.
#[derive(Clone, Copy)]
pub struct TypeAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> TypeAttribute<'c> {
    /// Creates a type attribute.
    pub fn new(r#type: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirTypeAttrGet(r#type.to_raw())) }
    }
}

attribute_traits!(TypeAttribute, is_type, "type");
