//! LLVM types

use crate::{context::Context, ir::Type, utility::into_raw_array};
use mlir_sys::{
    mlirLLVMArrayTypeGet, mlirLLVMFunctionTypeGet, mlirLLVMPointerTypeGet,
    mlirLLVMStructTypeLiteralGet, mlirLLVMVoidTypeGet,
};

// TODO Check if the `llvm` dialect is loaded on use of those functions.

/// Creates an LLVM array type.
pub fn array(r#type: Type, len: u32) -> Type {
    unsafe { Type::from_raw(mlirLLVMArrayTypeGet(r#type.to_raw(), len)) }
}

/// Creates an LLVM function type.
pub fn function<'c>(
    result: Type<'c>,
    arguments: &[Type<'c>],
    variadic_arguments: bool,
) -> Type<'c> {
    unsafe {
        Type::from_raw(mlirLLVMFunctionTypeGet(
            result.to_raw(),
            arguments.len() as isize,
            into_raw_array(arguments.iter().map(|argument| argument.to_raw()).collect()),
            variadic_arguments,
        ))
    }
}

/// Creates an LLVM pointer type.
pub fn pointer(r#type: Type, address_space: u32) -> Type {
    unsafe { Type::from_raw(mlirLLVMPointerTypeGet(r#type.to_raw(), address_space)) }
}

/// Creates an LLVM struct type.
pub fn r#struct<'c>(context: &'c Context, fields: &[Type<'c>], packed: bool) -> Type<'c> {
    unsafe {
        Type::from_raw(mlirLLVMStructTypeLiteralGet(
            context.to_raw(),
            fields.len() as isize,
            into_raw_array(fields.iter().map(|field| field.to_raw()).collect()),
            packed,
        ))
    }
}

/// Creates an LLVM void type.
pub fn void(context: &Context) -> Type {
    unsafe { Type::from_raw(mlirLLVMVoidTypeGet(context.to_raw())) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect;

    fn create_context() -> Context {
        let context = Context::new();

        dialect::Handle::llvm().register_dialect(&context);
        context.get_or_load_dialect("llvm");

        context
    }

    #[test]
    fn pointer() {
        let context = create_context();
        let i32 = Type::integer(&context, 32);

        assert_eq!(
            super::pointer(i32, 0),
            Type::parse(&context, "!llvm.ptr<i32>").unwrap()
        );
    }

    #[test]
    fn pointer_with_address_space() {
        let context = create_context();
        let i32 = Type::integer(&context, 32);

        assert_eq!(
            super::pointer(i32, 4),
            Type::parse(&context, "!llvm.ptr<i32, 4>").unwrap()
        );
    }

    #[test]
    fn void() {
        let context = create_context();

        assert_eq!(
            super::void(&context),
            Type::parse(&context, "!llvm.void").unwrap()
        );
    }

    #[test]
    fn array() {
        let context = create_context();
        let i32 = Type::integer(&context, 32);

        assert_eq!(
            super::array(i32, 4),
            Type::parse(&context, "!llvm.array<4xi32>").unwrap()
        );
    }

    #[test]
    fn function() {
        let context = create_context();
        let i8 = Type::integer(&context, 8);
        let i32 = Type::integer(&context, 32);
        let i64 = Type::integer(&context, 64);

        assert_eq!(
            super::function(i8, &[i32, i64], false),
            Type::parse(&context, "!llvm.func<i8 (i32, i64)>").unwrap()
        );
    }

    #[test]
    fn r#struct() {
        let context = create_context();
        let i32 = Type::integer(&context, 32);
        let i64 = Type::integer(&context, 64);

        assert_eq!(
            super::r#struct(&context, &[i32, i64], false),
            Type::parse(&context, "!llvm.struct<(i32, i64)>").unwrap()
        );
    }

    #[test]
    fn packed_struct() {
        let context = create_context();
        let i32 = Type::integer(&context, 32);
        let i64 = Type::integer(&context, 64);

        assert_eq!(
            super::r#struct(&context, &[i32, i64], true),
            Type::parse(&context, "!llvm.struct<packed (i32, i64)>").unwrap()
        );
    }
}
