//! Dialect conversion passes.

use super::Pass;
use mlir_sys::{
    mlirCreateConversionArithToLLVMConversionPass, mlirCreateConversionConvertControlFlowToLLVM,
    mlirCreateConversionConvertControlFlowToSPIRV, mlirCreateConversionConvertFuncToLLVM,
    mlirCreateConversionConvertMathToLLVM, mlirCreateConversionConvertMathToLibm,
    mlirCreateConversionConvertMathToSPIRV,
};

// TODO Unify a naming convention.

/// Creates a pass to convert the `arith` dialect to the `llvm` dialect.
pub fn convert_arithmetic_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionArithToLLVMConversionPass)
}

/// Creates a pass to convert the `cf` dialect to the `llvm` dialect.
pub fn convert_scf_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertControlFlowToLLVM)
}

/// Creates a pass to convert the `func` dialect to the `llvm` dialect.
pub fn convert_func_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertFuncToLLVM)
}

/// Creates a pass to convert the `math` dialect to the `llvm` dialect.
pub fn convert_math_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertMathToLLVM)
}

/// Creates a pass to convert the `cf` dialect to the `spirv` dialect.
pub fn convert_scf_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertControlFlowToSPIRV)
}

/// Creates a pass to convert the `math` dialect to the `spirv` dialect.
pub fn convert_math_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertMathToSPIRV)
}

/// Creates a pass to convert the `math` dialect to the `libm` dialect.
pub fn convert_math_to_libm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertMathToLibm)
}
