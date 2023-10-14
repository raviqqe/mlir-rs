use super::{Attribute, AttributeLike};
use crate::{Context, Error, StringRef};
use mlir_sys::{mlirStringAttrGet, MlirAttribute};

/// A string attribute.
#[derive(Clone, Copy)]
pub struct StringAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> StringAttribute<'c> {
    /// Creates a string attribute.
    pub fn new(context: &'c Context, string: &str) -> Self {
        unsafe {
            Self::from_raw(mlirStringAttrGet(
                context.to_raw(),
                StringRef::from_str(context, string).to_raw(),
            ))
        }
    }
}

attribute_traits!(StringAttribute, is_string, "string");
