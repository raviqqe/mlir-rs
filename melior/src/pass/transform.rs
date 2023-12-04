//! Transform passes.

melior_macro::passes!(
    "Transforms",
    [
        mlirCreateTransformsCSE,
        mlirCreateTransformsCanonicalizer,
        mlirCreateTransformsControlFlowSink,
        mlirCreateTransformsGenerateRuntimeVerification,
        mlirCreateTransformsInliner,
        mlirCreateTransformsLocationSnapshot,
        mlirCreateTransformsLoopInvariantCodeMotion,
        mlirCreateTransformsPrintOpStats,
        mlirCreateTransformsSCCP,
        mlirCreateTransformsStripDebugInfo,
        mlirCreateTransformsSymbolDCE,
        mlirCreateTransformsSymbolPrivatize,
        mlirCreateTransformsTopologicalSort,
        mlirCreateTransformsViewOpGraph,
    ]
);
