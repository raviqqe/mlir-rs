use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{mlirAttributeDump, mlirAttributeGetContext, mlirAttributeParseGet, MlirAttribute};
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
    pub fn parse(context: &Context, source: &str) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirAttributeParseGet(
                context.to_raw(),
                StringRef::from(source).to_raw(),
            ))
        }
    }

    /// Gets a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAttributeGetContext(self.raw)) }
    }

    /// Dumps an attribute.
    pub fn dump(&self) {
        unsafe { mlirAttributeDump(self.raw) }
    }

    unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    unsafe fn from_option_raw(raw: MlirAttribute) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }

    pub(crate) unsafe fn to_raw(self) -> MlirAttribute {
        self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        for attribute in ["unit", "i32", r#""foo""#] {
            assert!(Attribute::parse(&Context::new(), attribute).is_some());
        }
    }

    #[test]
    fn parse_none() {
        assert!(Attribute::parse(&Context::new(), "z").is_none());
    }

    #[test]
    fn context() {
        Attribute::parse(&Context::new(), "unit").unwrap().context();
    }
}
