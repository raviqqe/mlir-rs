//! Sparse tensor passes.

melior_macro::passes!(
    "SparseTensor",
    [
        mlirCreateSparseTensorLowerForeachToSCF,
        mlirCreateSparseTensorLowerSparseOpsToForeach,
        mlirCreateSparseTensorPreSparsificationRewrite,
        mlirCreateSparseTensorSparseBufferRewrite,
        mlirCreateSparseTensorSparseGPUCodegen,
        mlirCreateSparseTensorSparseReinterpretMap,
        mlirCreateSparseTensorSparseTensorCodegen,
        mlirCreateSparseTensorSparseTensorConversionPass,
        mlirCreateSparseTensorSparseVectorization,
        mlirCreateSparseTensorSparsificationAndBufferization,
        mlirCreateSparseTensorSparsificationPass,
        mlirCreateSparseTensorStageSparseOperations,
        mlirCreateSparseTensorStorageSpecifierToLLVM,
    ]
);
