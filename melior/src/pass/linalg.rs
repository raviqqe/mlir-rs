//! Linalg passes.

melior_macro::conversion_passes!(
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
