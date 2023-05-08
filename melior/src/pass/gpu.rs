//! GPU passes.

melior_macro::gpu_passes!(
    // spell-checker: disable-next-line
    mlirCreateGPUGPULowerMemorySpaceAttributesPass,
    mlirCreateGPUGpuAsyncRegionPass,
    mlirCreateGPUGpuKernelOutlining,
    mlirCreateGPUGpuLaunchSinkIndexComputations,
    mlirCreateGPUGpuMapParallelLoopsPass,
);
