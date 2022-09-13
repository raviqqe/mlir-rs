use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{mlirTypeGetContext, mlirTypeParseGet, MlirType};
use std::marker::PhantomData;

// Types are always values but their internal storage is owned by contexts.
#[derive(Clone, Copy, Debug)]
pub struct Type<'c> {
    raw: MlirType,
    _context: PhantomData<&'c Context>,
}

impl<'c> Type<'c> {
    pub fn parse(context: &Context, source: &str) -> Self {
        Self {
            raw: unsafe { mlirTypeParseGet(context.to_raw(), StringRef::from(source).to_raw()) },
            _context: Default::default(),
        }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirTypeGetContext(self.raw)) }
    }

    pub(crate) unsafe fn to_raw(self) -> MlirType {
        self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Type::parse(&Context::new(), "f32");
    }

    #[test]
    fn context() {
        Type::parse(&Context::new(), "i8").context();
    }
}
