use crate::{
    context::{Context, ContextRef},
    utility,
};
use std::marker::PhantomData;

pub struct Type<'c> {
    r#type: mlir_sys::MlirType,
    _context: PhantomData<&'c Context>,
}

impl<'c> Type<'c> {
    pub fn parse(context: &Context, source: &str) -> Self {
        Self {
            r#type: unsafe {
                mlir_sys::mlirTypeParseGet(context.to_raw(), utility::as_string_ref(source))
            },
            _context: Default::default(),
        }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlir_sys::mlirTypeGetContext(self.r#type)) }
    }

    pub(crate) unsafe fn to_raw(&self) -> mlir_sys::MlirType {
        self.r#type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Type::parse(&Context::new(), "foo");
    }

    #[test]
    fn context() {
        Type::parse(&Context::new(), "foo").context();
    }
}
