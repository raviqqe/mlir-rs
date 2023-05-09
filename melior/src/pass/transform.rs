//! Transform passes.

melior_macro::transform_passes!(
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
);
