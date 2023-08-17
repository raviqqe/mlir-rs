//! Experimental dialect operations and their builders generated automatically
//! from TableGen files.

melior_macro::dialect! {
    name: "affine",
    tablegen: r#"include "mlir/Dialect/Affine/IR/AffineOps.td""#
}
melior_macro::dialect! {
    name: "amdgpu",
    tablegen: r#"include "mlir/Dialect/AMDGPU/AMDGPU.td""#
}
melior_macro::dialect! {
    name: "arith",
    tablegen: r#"include "mlir/Dialect/Arith/IR/ArithOps.td""#
}
melior_macro::dialect! {
    name: "arm_neon",
    tablegen: r#"include "mlir/Dialect/ArmNeon/ArmNeon.td""#
}
melior_macro::dialect! {
    name: "arm_sve",
    tablegen: r#"include "mlir/Dialect/ArmSVE/ArmSVE.td""#
}
melior_macro::dialect! {
    name: "async",
    tablegen: r#"include "mlir/Dialect/Async/IR/AsyncOps.td""#
}
melior_macro::dialect! {
    name: "bufferization",
    tablegen: r#"include "mlir/Dialect/Bufferization/IR/BufferizationOps.td""#
}
melior_macro::dialect! {
    name: "cf",
    tablegen: r#"include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.td""#
}
melior_macro::dialect! {
    name: "func",
    tablegen: r#"include "mlir/Dialect/Func/IR/FuncOps.td""#
}
melior_macro::dialect! {
    name: "index",
    tablegen: r#"include "mlir/Dialect/Index/IR/IndexOps.td""#
}
melior_macro::dialect! {
    name: "llvm",
    // spell-checker: disable-next-line
    tablegen: r#"include "mlir/Dialect/LLVMIR/LLVMOps.td""#
}
melior_macro::dialect! {
    name: "memref",
    tablegen: r#"include "mlir/Dialect/MemRef/IR/MemRefOps.td""#
}
melior_macro::dialect! {
    name: "scf",
    tablegen: r#"include "mlir/Dialect/SCF/IR/SCFOps.td""#
}
melior_macro::dialect! {
    name: "pdl",
    tablegen: r#"include "mlir/Dialect/PDL/IR/PDLOps.td""#
}
melior_macro::dialect! {
    name: "pdl_interp",
    tablegen: r#"include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.td""#
}
melior_macro::dialect! {
    name: "math",
    tablegen: r#"include "mlir/Dialect/Math/IR/MathOps.td""#
}
melior_macro::dialect! {
    name: "gpu",
    tablegen: r#"include "mlir/Dialect/GPU/IR/GPUOps.td""#
}
melior_macro::dialect! {
    name: "linalg",
    tablegen: r#"include "mlir/Dialect/Linalg/IR/LinalgOps.td""#
}
melior_macro::dialect! {
    name: "quant",
    tablegen: r#"include "mlir/Dialect/Quant/QuantOps.td""#
}
melior_macro::dialect! {
    name: "shape",
    tablegen: r#"include "mlir/Dialect/Shape/IR/ShapeOps.td""#
}
melior_macro::dialect! {
    name: "sparse_tensor",
    tablegen: r#"include "mlir/Dialect/SparseTensor/IR/SparseTensorOps.td""#
}
melior_macro::dialect! {
    name: "tensor",
    tablegen: r#"include "mlir/Dialect/Tensor/IR/TensorOps.td""#
}
melior_macro::dialect! {
    name: "tosa",
    tablegen: r#"include "mlir/Dialect/Tosa/IR/TosaOps.td""#
}
melior_macro::dialect! {
    name: "transform",
    tablegen: r#"include "mlir/Dialect/Transform/IR/TransformOps.td""#
}
melior_macro::dialect! {
    name: "vector",
    tablegen: r#"include "mlir/Dialect/Vector/IR/VectorOps.td""#
}
