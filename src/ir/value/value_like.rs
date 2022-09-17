use super::Type;
use mlir_sys::{
    mlirValueDump, mlirValueGetType, mlirValueIsABlockArgument, mlirValueIsAOpResult, MlirValue,
};

/// A value-like trait.
pub trait ValueLike {
    /// Converts a raw value into a value.
    ///
    /// # Safety
    ///
    /// This function might create invalid values if raw values do not meet certain conditions.
    unsafe fn from_raw(value: MlirValue) -> Self;

    /// Converts a value into a raw value.
    ///
    /// # Safety
    ///
    /// This function might create invalid values if raw values do not meet certain conditions.
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
