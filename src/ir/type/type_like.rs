use super::Id;
use crate::context::ContextRef;
use mlir_sys::{
    mlirTypeDump, mlirTypeGetContext, mlirTypeGetTypeID, mlirTypeIsABF16, mlirTypeIsAF16,
    mlirTypeIsAF32, mlirTypeIsAF64, mlirTypeIsAFunction, mlirTypeIsATuple, mlirTypeIsAVector,
    MlirType,
};

pub trait TypeLike<'c> {
    /// Converts a type into a raw type.
    fn to_raw(&self) -> MlirType;

    /// Gets a context.
    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirTypeGetContext(self.to_raw())) }
    }

    /// Gets an ID.
    fn id(&self) -> Id {
        unsafe { Id::from_raw(mlirTypeGetTypeID(self.to_raw())) }
    }

    /// Returns `true` if a type is bfloat16.
    fn is_bfloat16(&self) -> bool {
        unsafe { mlirTypeIsABF16(self.to_raw()) }
    }

    /// Returns `true` if a type is float16.
    fn is_float16(&self) -> bool {
        unsafe { mlirTypeIsAF16(self.to_raw()) }
    }

    /// Returns `true` if a type is float32.
    fn is_float32(&self) -> bool {
        unsafe { mlirTypeIsAF32(self.to_raw()) }
    }

    /// Returns `true` if a type is float64.
    fn is_float64(&self) -> bool {
        unsafe { mlirTypeIsAF64(self.to_raw()) }
    }

    /// Returns `true` if a type is a function.
    fn is_function(&self) -> bool {
        unsafe { mlirTypeIsAFunction(self.to_raw()) }
    }

    /// Returns `true` if a type is a tuple.
    fn is_tuple(&self) -> bool {
        unsafe { mlirTypeIsATuple(self.to_raw()) }
    }

    /// Returns `true` if a type is a vector.
    fn is_vector(&self) -> bool {
        unsafe { mlirTypeIsAVector(self.to_raw()) }
    }

    /// Dumps a type.
    fn dump(&self) {
        unsafe { mlirTypeDump(self.to_raw()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{r#type::Function, Type},
        Context,
    };

    #[test]
    fn context() {
        Type::parse(&Context::new(), "i8").unwrap().context();
    }

    #[test]
    fn id() {
        let context = Context::new();

        assert_eq!(Type::index(&context).id(), Type::index(&context).id());
    }

    #[test]
    fn is_bfloat16() {
        let context = Context::new();

        assert!(Function::new(&context, &[], &[]).is_function());
    }

    #[test]
    fn is_function() {
        let context = Context::new();

        assert!(Function::new(&context, &[], &[]).is_function());
    }

    #[test]
    fn is_vector() {
        let context = Context::new();

        assert!(Type::vector(&[42], Type::integer(&context, 32)).is_vector());
    }

    #[test]
    fn dump() {
        Type::index(&Context::new()).dump();
    }
}
