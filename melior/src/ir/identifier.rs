use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{
    mlirIdentifierEqual, mlirIdentifierGet, mlirIdentifierGetContext, mlirIdentifierStr,
    MlirIdentifier,
};
use std::marker::PhantomData;

/// An identifier.
#[derive(Clone, Copy, Debug)]
pub struct Identifier<'c> {
    raw: MlirIdentifier,
    _context: PhantomData<&'c Context>,
}

impl<'c> Identifier<'c> {
    /// Creates an identifier.
    pub fn new(context: &Context, name: &str) -> Self {
        unsafe {
            Self::from_raw(mlirIdentifierGet(
                context.to_raw(),
                StringRef::from(name).to_raw(),
            ))
        }
    }

    /// Gets a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirIdentifierGetContext(self.raw)) }
    }

    /// Converts an identifier into a string reference.
    pub fn as_string_ref(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirIdentifierStr(self.raw)) }
    }

    pub(crate) unsafe fn from_raw(raw: MlirIdentifier) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    pub(crate) unsafe fn to_raw(self) -> MlirIdentifier {
        self.raw
    }
}

impl<'c> PartialEq for Identifier<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirIdentifierEqual(self.raw, other.raw) }
    }
}

impl<'c> Eq for Identifier<'c> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Identifier::new(&Context::new(), "foo");
    }

    #[test]
    fn context() {
        Identifier::new(&Context::new(), "foo").context();
    }

    #[test]
    fn equal() {
        let context = Context::new();

        assert_eq!(
            Identifier::new(&context, "foo"),
            Identifier::new(&context, "foo")
        );
    }

    #[test]
    fn not_equal() {
        let context = Context::new();

        assert_ne!(
            Identifier::new(&context, "foo"),
            Identifier::new(&context, "bar")
        );
    }
}
