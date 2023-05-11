use super::{Attribute, AttributeLike};
use crate::Error;
use mlir_sys::{mlirFlatSymbolRefAttrGet, MlirAttribute};

/// A flat symbol ref attribute.
#[derive(Clone, Copy)]
pub struct FlatSymbolRefAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> FlatSymbolRefAttribute<'c> {
    /// Creates an flat symbol ref attribute.
    pub fn new(context: &'c Context, symbol: &str) -> Self {
        unsafe { Self::from_raw(mlirFlatSymbolRefAttrGet(r#type.to_raw(), integer)) }
    }
}

attribute_traits!(FlatSymbolRefAttribute, is_integer, "integer");
