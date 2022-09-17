use super::Type;
use mlir_sys::{
    mlirValueDump, mlirValueGetType, mlirValueIsABlockArgument, mlirValueIsAOpResult, MlirValue,
};

/// A value-like trait.
pub trait ValueLike {
    unsafe fn from_raw(value: MlirValue) -> Self;

    unsafe fn to_raw(&self) -> MlirValue;

    /// Gets a type.
    fn r#type(&self) -> Type {
        unsafe { Type::from_raw(mlirValueGetType(self.to_raw())) }
    }

    /// Returns `true` if a value is a block argument.
    fn is_block_argument(&self) -> bool {
        unsafe { mlirValueIsABlockArgument(self.to_raw()) }
    }

    /// Returns `true` if a value is an operation result.
    fn is_operation_result(&self) -> bool {
        unsafe { mlirValueIsAOpResult(self.to_raw()) }
    }

    /// Dumps a value.
    fn dump(&self) {
        unsafe { mlirValueDump(self.to_raw()) }
    }
}
