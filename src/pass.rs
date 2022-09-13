use mlir_sys::{
    mlirCreateConversionConvertArithmeticToLLVM, mlirCreateConversionConvertFuncToLLVM, MlirPass,
};

/// A pass.
pub struct Pass {
    raw: MlirPass,
}

impl Pass {
    /// Creates a pass to convert the `arith` dialect to the `llvm` dialect.
    pub fn convert_arithmetic_to_llvm() -> Self {
        Self::from_raw_fn(mlirCreateConversionConvertArithmeticToLLVM)
    }

    /// Creates a pass to convert the `func` dialect to the `llvm` dialect.
    pub fn convert_func_to_llvm() -> Self {
        Self::from_raw_fn(mlirCreateConversionConvertFuncToLLVM)
    }

    // TODO Add more passes.

    fn from_raw_fn(create_raw: unsafe extern "C" fn() -> MlirPass) -> Self {
        Self {
            raw: unsafe { create_raw() },
        }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirPass {
        self.raw
    }
}
