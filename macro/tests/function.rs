use melior_macro::conversion_passes;

conversion_passes!(
    mlirCreateConversionArithToLLVMConversionPass,
    mlirCreateConversionConvertControlFlowToLLVM,
    mlirCreateConversionConvertControlFlowToSPIRV,
);
