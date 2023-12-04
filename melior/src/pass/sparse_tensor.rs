//! Sparse tensor passes.

melior_macro::passes!(
    "SparseTensor",
    [
        mlirCreateSparseTensorPostSparsificationRewrite,
        mlirCreateSparseTensorPreSparsificationRewrite,
        mlirCreateSparseTensorSparseBufferRewrite,
        mlirCreateSparseTensorSparseTensorCodegen,
        mlirCreateSparseTensorSparseTensorConversionPass,
        mlirCreateSparseTensorSparseVectorization,
        mlirCreateSparseTensorSparsificationPass,
        mlirCreateSparseTensorStorageSpecifierToLLVM,
    ]
);
