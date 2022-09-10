use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{mlirTypeGetContext, mlirTypeParseGet, MlirType};
use std::marker::PhantomData;

// Types are always values but their internal storage is owned by contexts.
pub struct Type<'c> {
    r#type: MlirType,
    _parent: PhantomData<&'c Context>,
}

impl<'c> Type<'c> {
    pub fn parse(context: &Context, source: &str) -> Self {
        Self {
            r#type: unsafe { mlirTypeParseGet(context.to_raw(), StringRef::from(source).to_raw()) },
            _parent: Default::default(),
        }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirTypeGetContext(self.r#type)) }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirType {
        self.r#type
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
