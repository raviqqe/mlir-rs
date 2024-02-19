//! Experimental dialect operations and their builders generated automatically
//! from TableGen files.

#[doc(hidden)]
pub mod __private {
    pub struct Set;
    pub struct Unset;
}

melior_macro::dialect! {
    name: "affine",
    table_gen: r#"include "mlir/Dialect/Affine/IR/AffineOps.td""#
}
melior_macro::dialect! {
    name: "amdgpu",
    table_gen: r#"include "mlir/Dialect/AMDGPU/IR/AMDGPU.td""#
}
melior_macro::dialect! {
    name: "arith",
    table_gen: r#"include "mlir/Dialect/Arith/IR/ArithOps.td""#
}
melior_macro::dialect! {
    name: "arm_neon",
    table_gen: r#"include "mlir/Dialect/ArmNeon/ArmNeon.td""#
}
melior_macro::dialect! {
    name: "arm_sve",
    table_gen: r#"include "mlir/Dialect/ArmSVE/ArmSVE.td""#
}
melior_macro::dialect! {
    name: "async",
    table_gen: r#"include "mlir/Dialect/Async/IR/AsyncOps.td""#
}
melior_macro::dialect! {
    name: "bufferization",
    table_gen: r#"include "mlir/Dialect/Bufferization/IR/BufferizationOps.td""#
}
melior_macro::dialect! {
    name: "cf",
    table_gen: r#"include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.td""#
}
melior_macro::dialect! {
    name: "func",
    table_gen: r#"include "mlir/Dialect/Func/IR/FuncOps.td""#
}
melior_macro::dialect! {
    name: "index",
    table_gen: r#"include "mlir/Dialect/Index/IR/IndexOps.td""#
}
melior_macro::dialect! {
    name: "llvm",
    // spell-checker: disable-next-line
    table_gen: r#"include "mlir/Dialect/LLVMIR/LLVMOps.td" include "mlir/Dialect/LLVMIR/LLVMIntrinsicOps.td""#
}
melior_macro::dialect! {
    name: "memref",
    table_gen: r#"include "mlir/Dialect/MemRef/IR/MemRefOps.td""#
}
melior_macro::dialect! {
    name: "scf",
    table_gen: r#"include "mlir/Dialect/SCF/IR/SCFOps.td""#
}
melior_macro::dialect! {
    name: "pdl",
    table_gen: r#"include "mlir/Dialect/PDL/IR/PDLOps.td""#
}
melior_macro::dialect! {
    name: "pdl_interp",
    table_gen: r#"include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.td""#
}
melior_macro::dialect! {
    name: "math",
    table_gen: r#"include "mlir/Dialect/Math/IR/MathOps.td""#
}
melior_macro::dialect! {
    name: "gpu",
    table_gen: r#"include "mlir/Dialect/GPU/IR/GPUOps.td""#
}
melior_macro::dialect! {
    name: "linalg",
    table_gen: r#"include "mlir/Dialect/Linalg/IR/LinalgOps.td""#
}
melior_macro::dialect! {
    name: "quant",
    table_gen: r#"include "mlir/Dialect/Quant/QuantOps.td""#
}
melior_macro::dialect! {
    name: "shape",
    table_gen: r#"include "mlir/Dialect/Shape/IR/ShapeOps.td""#
}
melior_macro::dialect! {
    name: "sparse_tensor",
    table_gen: r#"include "mlir/Dialect/SparseTensor/IR/SparseTensorOps.td""#
}
melior_macro::dialect! {
    name: "tensor",
    table_gen: r#"include "mlir/Dialect/Tensor/IR/TensorOps.td""#
}
melior_macro::dialect! {
    name: "tosa",
    table_gen: r#"include "mlir/Dialect/Tosa/IR/TosaOps.td""#
}
melior_macro::dialect! {
    name: "transform",
    table_gen: r#"include "mlir/Dialect/Transform/IR/TransformOps.td""#
}
melior_macro::dialect! {
    name: "vector",
    table_gen: r#"include "mlir/Dialect/Vector/IR/VectorOps.td""#
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{self, func},
        ir::{
            attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
            r#type::{FunctionType, IntegerType},
            Block, Location, Module, Region, Type,
        },
        pass::{self, PassManager},
        test::create_test_context,
        Context,
    };

    fn convert_module<'c>(context: &'c Context, module: &mut Module<'c>) {
        let pass_manager = PassManager::new(context);

        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_index_to_llvm());
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());

        assert_eq!(pass_manager.run(module), Ok(()));
        assert!(module.as_operation().verify());
    }

    fn test_operation<'c>(
        name: &str,
        context: &'c Context,
        argument_types: &[Type<'c>],
        callback: impl FnOnce(&Block<'c>),
    ) {
        let location = Location::unknown(context);
        let mut module = Module::new(location);

        module.body().append_operation(func::func(
            context,
            StringAttribute::new(context, "foo"),
            TypeAttribute::new(FunctionType::new(context, argument_types, &[]).into()),
            {
                let block = Block::new(
                    &argument_types
                        .iter()
                        .copied()
                        .map(|r#type| (r#type, location))
                        .collect::<Vec<_>>(),
                );

                callback(&block);

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(name, module.as_operation());
    }

    #[test]
    fn compile_arith_addf() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let r#type = Type::float32(&context);

        test_operation("addf", &context, &[r#type, r#type], |block| {
            block.append_operation(
                arith::addf(
                    &context,
                    block.argument(0).unwrap().into(),
                    block.argument(0).unwrap().into(),
                    location,
                )
                .into(),
            );

            block.append_operation(func::r#return(&[], location));
        });
    }

    #[test]
    fn compile_arith_addf_builder_with_reverse_order() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let r#type = Type::float32(&context);

        test_operation("addf", &context, &[r#type, r#type], |block| {
            block.append_operation(
                arith::AddFOperationBuilder::new(&context, location)
                    .rhs(block.argument(1).unwrap().into())
                    .lhs(block.argument(0).unwrap().into())
                    .into(),
            );

            block.append_operation(func::r#return(&[], location));
        });
    }

    #[test]
    fn compile_llvm_alloca() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let integer_type = IntegerType::new(&context, 64).into();

        test_operation("alloc", &context, &[integer_type], |block| {
            let alloca_size = block.argument(0).unwrap().into();

            block.append_operation(
                llvm::alloca(
                    &context,
                    dialect::llvm::r#type::pointer(integer_type.into(), 0),
                    alloca_size,
                    location,
                )
                .into(),
            );

            block.append_operation(func::r#return(&[], location));
        });
    }

    #[test]
    fn compile_llvm_alloca_builder() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = dialect::llvm::r#type::opaque_pointer(&context);

        test_operation("alloc_builder", &context, &[integer_type], |block| {
            let alloca_size = block.argument(0).unwrap().into();

            block.append_operation(
                llvm::AllocaOperationBuilder::new(&context, location)
                    .alignment(IntegerAttribute::new(8, integer_type.into()))
                    .elem_type(TypeAttribute::new(integer_type.into()))
                    .array_size(alloca_size)
                    .res(ptr_type)
                    .build()
                    .into(),
            );

            block.append_operation(func::r#return(&[], location));
        });
    }
}
