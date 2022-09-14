use crate::{
    context::{Context, ContextRef},
    error::Error,
    string_ref::StringRef,
    utility::into_raw_array,
};
use mlir_sys::{
    mlirFunctionTypeGet, mlirFunctionTypeGetInput, mlirFunctionTypeGetNumInputs,
    mlirFunctionTypeGetNumResults, mlirFunctionTypeGetResult, mlirIntegerTypeGet,
    mlirIntegerTypeSignedGet, mlirIntegerTypeUnsignedGet, mlirLLVMArrayTypeGet,
    mlirLLVMFunctionTypeGet, mlirLLVMPointerTypeGet, mlirLLVMStructTypeLiteralGet,
    mlirLLVMVoidTypeGet, mlirTypeDump, mlirTypeEqual, mlirTypeGetContext, mlirTypeIsAFunction,
    mlirTypeParseGet, mlirTypePrint, MlirStringRef, MlirType,
};
use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
};

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

    /// Gets an input of a function type.
    pub fn input(&self, position: usize) -> Result<Option<Self>, Error> {
        unsafe {
            if !mlirTypeIsAFunction(self.raw) {
                return Err(Error::FunctionExpected(*self));
            }

            Ok(Self::from_option_raw(mlirFunctionTypeGetInput(
                self.raw,
                position as isize,
            )))
        }
    }

    /// Gets a result of a function type.
    pub fn result(&self, position: usize) -> Result<Option<Self>, Error> {
        unsafe {
            if !mlirTypeIsAFunction(self.raw) {
                return Err(Error::FunctionExpected(*self));
            }

            Ok(Self::from_option_raw(mlirFunctionTypeGetResult(
                self.raw,
                position as isize,
            )))
        }
    }

    /// Gets a number of inputs of a function type.
    pub fn input_count(&self) -> Result<usize, Error> {
        unsafe {
            if !mlirTypeIsAFunction(self.raw) {
                return Err(Error::FunctionExpected(*self));
            }

            Ok(mlirFunctionTypeGetNumInputs(self.raw) as usize)
        }
    }

    /// Gets a number of results of a function type.
    pub fn result_count(&self) -> Result<usize, Error> {
        unsafe {
            if !mlirTypeIsAFunction(self.raw) {
                return Err(Error::FunctionExpected(*self));
            }

            Ok(mlirFunctionTypeGetNumResults(self.raw) as usize)
        }
    }

    /// Dumps a type.
    pub fn dump(&self) {
        unsafe { mlirTypeDump(self.raw) }
    }

    pub(crate) unsafe fn from_raw(raw: MlirType) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    pub(crate) unsafe fn from_option_raw(raw: MlirType) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
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

impl<'c> Display for Type<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe extern "C" fn callback(string: MlirStringRef, data: *mut c_void) {
            let data = &mut *(data as *mut (&mut Formatter, fmt::Result));
            let result = write!(data.0, "{}", StringRef::from_raw(string).as_str());

            if data.1.is_ok() {
                data.1 = result;
            }
        }

        unsafe {
            mlirTypePrint(self.raw, Some(callback), &mut data as *mut _ as *mut c_void);
        }

        data.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dialect_handle::DialectHandle, error::Error};

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
    fn display() {
        let context = Context::new();

        assert_eq!(Type::integer(&context, 42).to_string(), "i42");
    }

    mod function {
        use super::*;

        #[test]
        fn function() {
            let context = Context::new();
            let integer = Type::integer(&context, 42);

            assert_eq!(
                Type::function(&context, &[integer, integer], &[integer]),
                Type::parse(&context, "(i42, i42) -> i42")
            );
        }

        #[test]
        fn multiple_results() {
            let context = Context::new();
            let integer = Type::integer(&context, 42);

            assert_eq!(
                Type::function(&context, &[], &[integer, integer]),
                Type::parse(&context, "() -> (i42, i42)")
            );
        }

        #[test]
        fn input() {
            let context = Context::new();
            let integer = Type::integer(&context, 42);

            assert_eq!(
                Type::function(&context, &[integer], &[]).input(0),
                Ok(Some(integer))
            );
        }

        #[test]
        fn input_with_non_function() {
            let context = Context::new();
            let r#type = Type::integer(&context, 42);

            assert_eq!(r#type.input(0), Err(Error::FunctionExpected(r#type)));
        }

        #[test]
        fn result() {
            let context = Context::new();
            let integer = Type::integer(&context, 42);

            assert_eq!(
                Type::function(&context, &[], &[integer]).result(0),
                Ok(Some(integer))
            );
        }

        #[test]
        fn result_with_non_function() {
            let context = Context::new();
            let r#type = Type::integer(&context, 42);

            assert_eq!(r#type.result(0), Err(Error::FunctionExpected(r#type)));
        }

        #[test]
        fn input_count() {
            let context = Context::new();
            let integer = Type::integer(&context, 42);

            assert_eq!(
                Type::function(&context, &[integer], &[]).input_count(),
                Ok(1)
            );
        }

        #[test]
        fn input_count_with_non_function() {
            let context = Context::new();
            let r#type = Type::integer(&context, 42);

            assert_eq!(r#type.input_count(), Err(Error::FunctionExpected(r#type)));
        }

        #[test]
        fn result_count() {
            let context = Context::new();
            let integer = Type::integer(&context, 42);

            assert_eq!(
                Type::function(&context, &[], &[integer]).result_count(),
                Ok(1)
            );
        }

        #[test]
        fn result_count_with_non_function() {
            let context = Context::new();
            let r#type = Type::integer(&context, 42);

            assert_eq!(r#type.result_count(), Err(Error::FunctionExpected(r#type)));
        }
    }

    mod llvm {
        use super::*;

        fn create_context() -> Context {
            let context = Context::new();

            DialectHandle::llvm().register_dialect(&context);
            context.get_or_load_dialect("llvm");

            context
        }

        #[test]
        fn pointer() {
            let context = create_context();
            let i32 = Type::integer(&context, 32);

            assert_eq!(
                Type::llvm_pointer(i32, 0),
                Type::parse(&context, "!llvm.ptr<i32>")
            );
        }

        #[test]
        fn pointer_with_address_space() {
            let context = create_context();
            let i32 = Type::integer(&context, 32);

            assert_eq!(
                Type::llvm_pointer(i32, 4),
                Type::parse(&context, "!llvm.ptr<i32, 4>")
            );
        }

        #[test]
        fn void() {
            let context = create_context();

            assert_eq!(
                Type::llvm_void(&context),
                Type::parse(&context, "!llvm.void")
            );
        }

        #[test]
        fn array() {
            let context = create_context();
            let i32 = Type::integer(&context, 32);

            assert_eq!(
                Type::llvm_array(i32, 4),
                Type::parse(&context, "!llvm.array<4xi32>")
            );
        }

        #[test]
        fn function() {
            let context = create_context();
            let i8 = Type::integer(&context, 8);
            let i32 = Type::integer(&context, 32);
            let i64 = Type::integer(&context, 64);

            assert_eq!(
                Type::llvm_function(i8, &[i32, i64], false),
                Type::parse(&context, "!llvm.func<i8 (i32, i64)>")
            );
        }

        #[test]
        fn r#struct() {
            let context = create_context();
            let i32 = Type::integer(&context, 32);
            let i64 = Type::integer(&context, 64);

            assert_eq!(
                Type::llvm_struct(&context, &[i32, i64], false),
                Type::parse(&context, "!llvm.struct<(i32, i64)>")
            );
        }

        #[test]
        fn packed_struct() {
            let context = create_context();
            let i32 = Type::integer(&context, 32);
            let i64 = Type::integer(&context, 64);

            assert_eq!(
                Type::llvm_struct(&context, &[i32, i64], true),
                Type::parse(&context, "!llvm.struct<packed (i32, i64)>")
            );
        }
    }
}
