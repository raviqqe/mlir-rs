//! `async` dialect passes.

melior_macro::async_passes!(
    mlirCreateAsyncAsyncFuncToAsyncRuntime,
    mlirCreateAsyncAsyncParallelFor,
    mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting,
    mlirCreateAsyncAsyncRuntimeRefCounting,
    mlirCreateAsyncAsyncRuntimeRefCountingOpt,
    mlirCreateAsyncAsyncToAsyncRuntime,
);
