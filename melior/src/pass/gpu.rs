//! Dialect conversion passes.

melior_macro::conversion_passes!(
    mlirCreateGPUGPULowerMemorySpaceAttributesPass,
    mlirCreateGPUGpuAsyncRegionPass,
    mlirCreateGPUGpuKernelOutlining,
    mlirCreateGPUGpuLaunchSinkIndexComputations,
    mlirCreateGPUGpuMapParallelLoopsPass,
);
