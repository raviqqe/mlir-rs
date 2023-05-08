//! Sparse tensor passes.

melior_macro::sparse_tensor_passes!(
    mlirCreateSparseTensorPostSparsificationRewrite,
    mlirCreateSparseTensorPreSparsificationRewrite,
    mlirCreateSparseTensorSparseBufferRewrite,
    mlirCreateSparseTensorSparseTensorCodegen,
    mlirCreateSparseTensorSparseTensorConversionPass,
    mlirCreateSparseTensorSparseVectorization,
    mlirCreateSparseTensorSparsificationPass,
    mlirCreateSparseTensorStorageSpecifierToLLVM,
);
