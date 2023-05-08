//! Dialect conversion passes.

melior_macro::conversion_passes!(
    mlirCreateConversionArithToLLVMConversionPass,
    mlirCreateConversionConvertControlFlowToLLVM,
    mlirCreateConversionConvertControlFlowToSPIRV,
    mlirCreateConversionConvertFuncToLLVM,
    mlirCreateConversionConvertMathToLLVM,
    mlirCreateConversionConvertMathToLibm,
    mlirCreateConversionConvertMathToSPIRV
);
