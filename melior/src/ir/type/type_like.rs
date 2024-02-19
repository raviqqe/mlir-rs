use super::TypeId;
use crate::context::ContextRef;
use mlir_sys::{mlirTypeDump, mlirTypeGetContext, mlirTypeGetTypeID, MlirType};

/// Trait for type-like types.
pub trait TypeLike<'c> {
    /// Converts a type into a raw object.
    fn to_raw(&self) -> MlirType;

    /// Returns a context.
    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirTypeGetContext(self.to_raw())) }
    }

    /// Returns an ID.
    fn id(&self) -> TypeId<'c> {
        unsafe { TypeId::from_raw(mlirTypeGetTypeID(self.to_raw())) }
    }

    /// Dumps a type.
    fn dump(&self) {
        unsafe { mlirTypeDump(self.to_raw()) }
    }

    melior_macro::type_check_functions!(
        mlirTypeIsAAnyQuantizedType,
        mlirTypeIsABF16,
        mlirTypeIsACalibratedQuantizedType,
        mlirTypeIsAComplex,
        mlirTypeIsAF16,
        mlirTypeIsAF32,
        mlirTypeIsAF64,
        mlirTypeIsAFloat8E4M3FN,
        mlirTypeIsAFloat8E5M2,
        mlirTypeIsAFunction,
        mlirTypeIsAIndex,
        mlirTypeIsAInteger,
        mlirTypeIsAMemRef,
        mlirTypeIsANone,
        mlirTypeIsAOpaque,
        mlirTypeIsAPDLAttributeType,
        mlirTypeIsAPDLOperationType,
        mlirTypeIsAPDLRangeType,
        mlirTypeIsAPDLType,
        mlirTypeIsAPDLTypeType,
        mlirTypeIsAPDLValueType,
        mlirTypeIsAQuantizedType,
        mlirTypeIsARankedTensor,
        mlirTypeIsAShaped,
        mlirTypeIsATensor,
        mlirTypeIsATransformAnyOpType,
        mlirTypeIsATransformOperationType,
        mlirTypeIsATuple,
        mlirTypeIsAUniformQuantizedPerAxisType,
        mlirTypeIsAUniformQuantizedType,
        mlirTypeIsAUnrankedMemRef,
        mlirTypeIsAUnrankedTensor,
        mlirTypeIsAVector,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            r#type::{FunctionType, IntegerType},
            Type,
        },
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
    fn is_integer() {
        let context = Context::new();

        assert!(IntegerType::new(&context, 64).is_integer());
    }

    #[test]
    fn is_index() {
        let context = Context::new();

        assert!(Type::index(&context).is_index());
    }

    #[test]
    fn is_bfloat16() {
        let context = Context::new();

        assert!(FunctionType::new(&context, &[], &[]).is_function());
    }

    #[test]
    fn is_function() {
        let context = Context::new();

        assert!(FunctionType::new(&context, &[], &[]).is_function());
    }

    #[test]
    fn is_vector() {
        let context = Context::new();

        assert!(Type::vector(&[42], Type::index(&context)).is_vector());
    }

    #[test]
    fn dump() {
        Type::index(&Context::new()).dump();
    }
}
