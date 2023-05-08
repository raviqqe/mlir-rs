//! Linalg passes.

melior_macro::linalg_passes!(
    mlirCreateLinalgConvertElementwiseToLinalg,
    mlirCreateLinalgLinalgBufferize,
    mlirCreateLinalgLinalgDetensorize,
    mlirCreateLinalgLinalgElementwiseOpFusion,
    mlirCreateLinalgLinalgFoldUnitExtentDims,
    mlirCreateLinalgLinalgGeneralization,
    mlirCreateLinalgLinalgInlineScalarOperands,
    mlirCreateLinalgLinalgLowerToAffineLoops,
    mlirCreateLinalgLinalgLowerToLoops,
    mlirCreateLinalgLinalgLowerToParallelLoops,
    mlirCreateLinalgLinalgNamedOpConversion,
);
