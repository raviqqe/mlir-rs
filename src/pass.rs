use mlir_sys::{
    mlirCreateConversionConvertArithmeticToLLVM, mlirCreateConversionConvertControlFlowToLLVM,
    mlirCreateConversionConvertControlFlowToSPIRV, mlirCreateConversionConvertFuncToLLVM,
    mlirCreateConversionConvertMathToLLVM, mlirCreateConversionConvertMathToLibm,
    mlirCreateConversionConvertMathToSPIRV, MlirPass,
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

    /// Creates a pass to convert the `cf` dialect to the `llvm` dialect.
    pub fn convert_scf_to_llvm() -> Self {
        Self::from_raw_fn(mlirCreateConversionConvertControlFlowToLLVM)
    }

    /// Creates a pass to convert the `func` dialect to the `llvm` dialect.
    pub fn convert_func_to_llvm() -> Self {
        Self::from_raw_fn(mlirCreateConversionConvertFuncToLLVM)
    }

    /// Creates a pass to convert the `math` dialect to the `llvm` dialect.
    pub fn convert_math_to_llvm() -> Self {
        Self::from_raw_fn(mlirCreateConversionConvertMathToLLVM)
    }

    /// Creates a pass to convert the `cf` dialect to the `spirv` dialect.
    pub fn convert_scf_to_spirv() -> Self {
        Self::from_raw_fn(mlirCreateConversionConvertControlFlowToSPIRV)
    }

    /// Creates a pass to convert the `math` dialect to the `spirv` dialect.
    pub fn convert_math_to_spirv() -> Self {
        Self::from_raw_fn(mlirCreateConversionConvertMathToSPIRV)
    }

    /// Creates a pass to convert the `math` dialect to the `libm` dialect.
    pub fn convert_math_to_libm() -> Self {
        Self::from_raw_fn(mlirCreateConversionConvertMathToLibm)
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
