use crate::{
    context::{Context, ContextRef},
    utility::as_string_ref,
};
use mlir_sys::{mlirTypeGetContext, mlirTypeParseGet, MlirType};
use std::marker::PhantomData;

// Types are always references.
pub struct Type<'a> {
    r#type: MlirType,
    _parent: PhantomData<&'a MlirType>,
}

impl<'c> Type<'c> {
    pub fn parse(context: &Context, source: &str) -> Self {
        Self {
            r#type: unsafe { mlirTypeParseGet(context.to_raw(), as_string_ref(source)) },
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
