use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
    utility::into_raw_array,
};
use mlir_sys::{
    mlirFunctionTypeGet, mlirIntegerTypeGet, mlirIntegerTypeSignedGet, mlirIntegerTypeUnsignedGet,
    mlirLLVMArrayTypeGet, mlirLLVMFunctionTypeGet, mlirLLVMPointerTypeGet,
    mlirLLVMStructTypeLiteralGet, mlirLLVMVoidTypeGet, mlirTypeEqual, mlirTypeGetContext,
    mlirTypeParseGet, MlirType,
};
use std::marker::PhantomData;

/// A type.
// Types are always values but their internal storage is owned by contexts.
#[derive(Clone, Copy, Debug)]
pub struct Type<'c> {
    raw: MlirType,
    _context: PhantomData<&'c Context>,
}

impl<'c> Type<'c> {
    /// Parses a type.
    pub fn parse(context: &'c Context, source: &str) -> Self {
        unsafe {
            Self::from_raw(mlirTypeParseGet(
                context.to_raw(),
                StringRef::from(source).to_raw(),
            ))
        }
    }

    /// Creates an integer type.
    pub fn function(context: &'c Context, inputs: &[Type<'c>], results: &[Type<'c>]) -> Self {
        unsafe {
            Self::from_raw(mlirFunctionTypeGet(
                context.to_raw(),
                inputs.len() as isize,
                into_raw_array(inputs.iter().map(|r#type| r#type.to_raw()).collect()),
                results.len() as isize,
                into_raw_array(results.iter().map(|r#type| r#type.to_raw()).collect()),
            ))
        }
    }

    /// Creates an integer type.
    pub fn integer(context: &'c Context, bits: u32) -> Self {
        unsafe { Self::from_raw(mlirIntegerTypeGet(context.to_raw(), bits)) }
    }

    /// Creates a signed integer type.
    pub fn signed_integer(context: &'c Context, bits: u32) -> Self {
        unsafe { Self::from_raw(mlirIntegerTypeSignedGet(context.to_raw(), bits)) }
    }

    /// Creates an unsigned integer type.
    pub fn unsigned_integer(context: &'c Context, bits: u32) -> Self {
        unsafe { Self::from_raw(mlirIntegerTypeUnsignedGet(context.to_raw(), bits)) }
    }

    /// Creates an LLVM array type.
    // TODO Check if the `llvm` dialect is loaded.
    pub fn llvm_array(r#type: Type<'c>, len: u32) -> Self {
        unsafe { Self::from_raw(mlirLLVMArrayTypeGet(r#type.to_raw(), len)) }
    }

    /// Creates an LLVM function type.
    pub fn llvm_function(
        result: Type<'c>,
        arguments: &[Type<'c>],
        variadic_arguments: bool,
    ) -> Self {
        unsafe {
            Self::from_raw(mlirLLVMFunctionTypeGet(
                result.to_raw(),
                arguments.len() as isize,
                into_raw_array(arguments.iter().map(|argument| argument.to_raw()).collect()),
                variadic_arguments,
            ))
        }
    }

    /// Creates an LLVM pointer type.
    pub fn llvm_pointer(r#type: Self, address_space: u32) -> Self {
        unsafe { Self::from_raw(mlirLLVMPointerTypeGet(r#type.to_raw(), address_space)) }
    }

    /// Creates an LLVM struct type.
    pub fn llvm_struct(context: &'c Context, fields: &[Type<'c>], packed: bool) -> Self {
        unsafe {
            Self::from_raw(mlirLLVMStructTypeLiteralGet(
                context.to_raw(),
                fields.len() as isize,
                into_raw_array(fields.iter().map(|field| field.to_raw()).collect()),
                packed,
            ))
        }
    }

    /// Creates an LLVM void type.
    pub fn llvm_void(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirLLVMVoidTypeGet(context.to_raw())) }
    }

    /// Gets a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirTypeGetContext(self.raw)) }
    }

    pub(crate) unsafe fn from_raw(raw: MlirType) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
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
    use crate::dialect_handle::DialectHandle;

    #[test]
    fn new() {
        Type::parse(&Context::new(), "f32");
    }

    #[test]
    fn context() {
        Type::parse(&Context::new(), "i8").context();
    }

    #[test]
    fn integer() {
        let context = Context::new();

        assert_eq!(Type::integer(&context, 42), Type::parse(&context, "i42"));
    }

    #[test]
    fn signed_integer() {
        let context = Context::new();

        assert_eq!(
            Type::signed_integer(&context, 42),
            Type::parse(&context, "si42")
        );
    }

    #[test]
    fn unsigned_integer() {
        let context = Context::new();

        assert_eq!(
            Type::unsigned_integer(&context, 42),
            Type::parse(&context, "ui42")
        );
    }

    #[test]
    fn create_llvm_types() {
        let context = Context::new();

        DialectHandle::llvm().register_dialect(&context);
        context.get_or_load_dialect("llvm");

        let i8 = Type::integer(&context, 8);
        let i32 = Type::integer(&context, 32);
        let i64 = Type::integer(&context, 64);

        assert_eq!(
            Type::llvm_pointer(i32, 0),
            Type::parse(&context, "!llvm.ptr<i32>")
        );

        assert_eq!(
            Type::llvm_pointer(i32, 4),
            Type::parse(&context, "!llvm.ptr<i32, 4>")
        );

        assert_eq!(
            Type::llvm_void(&context),
            Type::parse(&context, "!llvm.void")
        );

        assert_eq!(
            Type::llvm_array(i32, 4),
            Type::parse(&context, "!llvm.array<4xi32>")
        );

        assert_eq!(
            Type::llvm_function(i8, &[i32, i64], false),
            Type::parse(&context, "!llvm.func<i8 (i32, i64)>")
        );

        assert_eq!(
            Type::llvm_struct(&context, &[i32, i64], false),
            Type::parse(&context, "!llvm.struct<(i32, i64)>")
        );
    }
}
