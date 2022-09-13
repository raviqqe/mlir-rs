use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{mlirAttributeGetContext, mlirAttributeParseGet, MlirAttribute};
use std::marker::PhantomData;

/// An attribute.
// Attributes are always values but their internal storage is owned by contexts.
#[derive(Clone, Copy, Debug)]
pub struct Attribute<'c> {
    raw: MlirAttribute,
    _context: PhantomData<&'c Context>,
}

impl<'c> Attribute<'c> {
    /// Parses an attribute.
    pub fn parse(context: &Context, source: &str) -> Self {
        Self {
            raw: unsafe {
                mlirAttributeParseGet(context.to_raw(), StringRef::from(source).to_raw())
            },
            _context: Default::default(),
        }
    }

    /// Gets a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAttributeGetContext(self.raw)) }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirAttribute {
        self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        for attribute in ["unit", "i32", r#""foo""#] {
            Attribute::parse(&Context::new(), attribute);
        }
    }

    #[test]
    fn context() {
        Attribute::parse(&Context::new(), "unit").context();
    }
}
