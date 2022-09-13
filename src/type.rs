use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{
    mlirIntegerTypeGet, mlirIntegerTypeSignedGet, mlirIntegerTypeUnsignedGet,
    mlirLLVMPointerTypeGet, mlirTypeEqual, mlirTypeGetContext, mlirTypeParseGet, MlirType,
};
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

    pub fn integer(context: &Context, bits: u32) -> Self {
        Self {
            raw: unsafe { mlirIntegerTypeGet(context.to_raw(), bits) },
            _context: Default::default(),
        }
    }

    pub fn signed_integer(context: &Context, bits: u32) -> Self {
        Self {
            raw: unsafe { mlirIntegerTypeSignedGet(context.to_raw(), bits) },
            _context: Default::default(),
        }
    }

    pub fn unsigned_integer(context: &Context, bits: u32) -> Self {
        Self {
            raw: unsafe { mlirIntegerTypeUnsignedGet(context.to_raw(), bits) },
            _context: Default::default(),
        }
    }

    pub fn llvm_pointer(r#type: Self, address_space: u32) -> Self {
        Self {
            raw: unsafe { mlirLLVMPointerTypeGet(r#type.to_raw(), address_space) },
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

impl<'c> PartialEq for Type<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirTypeEqual(self.raw, other.raw) }
    }
}

impl<'c> Eq for Type<'c> {}

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
