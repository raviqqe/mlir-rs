//! Dialect conversion passes.

melior_macro::conversion_passes!(
    mlirCreateAsyncAsyncFuncToAsyncRuntime,
    mlirCreateAsyncAsyncParallelFor,
    mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting,
    mlirCreateAsyncAsyncRuntimeRefCounting,
    mlirCreateAsyncAsyncRuntimeRefCountingOpt,
    mlirCreateAsyncAsyncToAsyncRuntime,
);
