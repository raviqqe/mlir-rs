//! Linalg passes.

melior_macro::passes!(
    "Linalg",
    [
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
    ]
);
