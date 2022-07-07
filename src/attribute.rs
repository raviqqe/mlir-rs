use crate::{
    context::{Context, ContextRef},
    utility,
};
use mlir_sys::{mlirAttributeParseGet, MlirAttribute};
use std::marker::PhantomData;

pub struct Attribute<'c> {
    attribute: MlirAttribute,
    _context: PhantomData<&'c Context>,
}

impl<'c> Attribute<'c> {
    pub fn parse(context: &Context, source: &str) -> Self {
        Self {
            attribute: unsafe {
                mlirAttributeParseGet(context.to_raw(), utility::as_string_ref(source))
            },
            _context: Default::default(),
        }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlir_sys::mlirAttributeGetContext(self.attribute)) }
    }

    pub(crate) unsafe fn to_raw(&self) -> mlir_sys::MlirAttribute {
        self.attribute
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        Attribute::parse(&Context::new(), "foo");
    }

    #[test]
    fn context() {
        Attribute::parse(&Context::new(), "foo").context();
    }
}
