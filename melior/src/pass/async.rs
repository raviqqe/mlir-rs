//! Async passes.

melior_macro::passes!(
    "Async",
    [
        mlirCreateAsyncAsyncFuncToAsyncRuntime,
        mlirCreateAsyncAsyncParallelFor,
        mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting,
        mlirCreateAsyncAsyncRuntimeRefCounting,
        mlirCreateAsyncAsyncRuntimeRefCountingOpt,
        mlirCreateAsyncAsyncToAsyncRuntime,
    ]
);
