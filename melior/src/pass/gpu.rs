//! GPU passes.

melior_macro::passes!(
    "GPU",
    [
        // spell-checker: disable-next-line
        mlirCreateGPUGpuAsyncRegionPass,
        mlirCreateGPUGpuKernelOutlining,
        mlirCreateGPUGpuLaunchSinkIndexComputations,
        mlirCreateGPUGpuMapParallelLoopsPass,
    ]
);
