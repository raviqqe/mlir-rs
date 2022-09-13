pub mod attribute;
pub mod block;
pub mod context;
pub mod dialect;
pub mod dialect_handle;
pub mod dialect_registry;
pub mod execution_engine;
pub mod identifier;
pub mod location;
pub mod logical_result;
pub mod module;
pub mod operation;
pub mod operation_pass_manager;
pub mod operation_state;
pub mod pass;
pub mod pass_manager;
pub mod region;
pub mod string_ref;
pub mod r#type;
pub mod utility;
pub mod value;

#[cfg(test)]
mod tests {
    use crate::{
        attribute::Attribute, block::Block, context::Context, dialect_registry::DialectRegistry,
        identifier::Identifier, location::Location, module::Module, operation::Operation,
        operation_state::OperationState, r#type::Type, region::Region,
        utility::register_all_dialects,
    };

    #[test]
    fn build_module() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(&*module.as_operation());
    }

    #[test]
    fn build_module_with_dialect() {
        let registry = DialectRegistry::new();
        let context = Context::new();
        context.append_dialect_registry(&registry);
        let module = Module::new(Location::unknown(&context));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(&*module.as_operation());
    }

    #[test]
    fn build_add() {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("func");
        context.get_or_load_dialect("memref");
        context.get_or_load_dialect("shape");
        context.get_or_load_dialect("scf");

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let r#type = Type::parse(&context, "memref<?xf32>");

        let function = {
            let function_region = Region::new();
            let function_block = Block::new(&[(r#type, location), (r#type, location)]);
            let index_type = Type::parse(&context, "index");

            let zero = function_block.append_operation(Operation::new(
                OperationState::new("arith.constant", location)
                    .add_results(&[index_type])
                    .add_attributes(&[(
                        Identifier::new(&context, "value"),
                        Attribute::parse(&context, "0 : index"),
                    )]),
            ));

            let dim = function_block.append_operation(Operation::new(
                OperationState::new("memref.dim", location)
                    .add_operands(&[function_block.argument(0).unwrap(), zero.result(0).unwrap()])
                    .add_results(&[index_type]),
            ));

            let loop_block = Block::new(&[]);
            loop_block.add_argument(index_type, location);

            let one = function_block.append_operation(Operation::new(
                OperationState::new("arith.constant", location)
                    .add_results(&[index_type])
                    .add_attributes(&[(
                        Identifier::new(&context, "value"),
                        Attribute::parse(&context, "1 : index"),
                    )]),
            ));

            {
                let f32_type = Type::parse(&context, "f32");

                let lhs = loop_block.append_operation(Operation::new(
                    OperationState::new("memref.load", location)
                        .add_operands(&[
                            function_block.argument(0).unwrap(),
                            loop_block.argument(0).unwrap(),
                        ])
                        .add_results(&[f32_type]),
                ));

                let rhs = loop_block.append_operation(Operation::new(
                    OperationState::new("memref.load", location)
                        .add_operands(&[
                            function_block.argument(1).unwrap(),
                            loop_block.argument(0).unwrap(),
                        ])
                        .add_results(&[f32_type]),
                ));

                let add = loop_block.append_operation(Operation::new(
                    OperationState::new("arith.addf", location)
                        .add_operands(&[lhs.result(0).unwrap(), rhs.result(0).unwrap()])
                        .add_results(&[f32_type]),
                ));

                loop_block.append_operation(Operation::new(
                    OperationState::new("memref.store", location).add_operands(&[
                        add.result(0).unwrap(),
                        function_block.argument(0).unwrap(),
                        loop_block.argument(0).unwrap(),
                    ]),
                ));

                loop_block
                    .append_operation(Operation::new(OperationState::new("scf.yield", location)));
            }

            function_block.append_operation(Operation::new({
                let loop_region = Region::new();

                loop_region.append_block(loop_block);

                OperationState::new("scf.for", location)
                    .add_operands(&[
                        zero.result(0).unwrap(),
                        dim.result(0).unwrap(),
                        one.result(0).unwrap(),
                    ])
                    .add_regions(vec![loop_region])
            }));

            function_block.append_operation(Operation::new(OperationState::new(
                "func.return",
                Location::unknown(&context),
            )));

            function_region.append_block(function_block);

            Operation::new(
                OperationState::new("func.func", Location::unknown(&context))
                    .add_attributes(&[
                        (
                            Identifier::new(&context, "function_type"),
                            Attribute::parse(&context, "(memref<?xf32>, memref<?xf32>) -> ()"),
                        ),
                        (
                            Identifier::new(&context, "sym_name"),
                            Attribute::parse(&context, "\"add\""),
                        ),
                    ])
                    .add_regions(vec![function_region]),
            )
        };

        module.body().insert_operation(0, function);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(&*module.as_operation());
    }

    #[test]
    fn create_llvm_types() {
        let context = Context::new();

        let i8 = Type::integer(&context, 8);
        let i32 = Type::integer(&context, 32);
        let i64 = Type::integer(&context, 64);

        let i32_pointer = Type::llvm_pointer(i32, 0);

        assert_eq!(i32_pointer, Type::parse(&context, "!llvm.ptr<i32>"));

        // const char *i32p4_text = "!llvm.ptr<i32, 4>";
        // MlirType i32p4 = mlirLLVMPointerTypeGet(i32, 4);
        // MlirType i32p4_ref = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i32p4_text));
        // // CHECK: !llvm.ptr<i32, 4>: 1
        // fprintf(stderr, "%s: %d\n", i32p4_text, mlirTypeEqual(i32p4, i32p4_ref));

        // const char *voidt_text = "!llvm.void";
        // MlirType voidt = mlirLLVMVoidTypeGet(ctx);
        // MlirType voidt_ref =
        //     mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(voidt_text));
        // // CHECK: !llvm.void: 1
        // fprintf(stderr, "%s: %d\n", voidt_text, mlirTypeEqual(voidt, voidt_ref));

        // const char *i32_4_text = "!llvm.array<4xi32>";
        // MlirType i32_4 = mlirLLVMArrayTypeGet(i32, 4);
        // MlirType i32_4_ref =
        //     mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i32_4_text));
        // // CHECK: !llvm.array<4xi32>: 1
        // fprintf(stderr, "%s: %d\n", i32_4_text, mlirTypeEqual(i32_4, i32_4_ref));

        // const char *i8_i32_i64_text = "!llvm.func<i8 (i32, i64)>";
        // const MlirType i32_i64_arr[] = {i32, i64};
        // MlirType i8_i32_i64 = mlirLLVMFunctionTypeGet(i8, 2, i32_i64_arr, false);
        // MlirType i8_i32_i64_ref =
        //     mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i8_i32_i64_text));
        // // CHECK: !llvm.func<i8 (i32, i64)>: 1
        // fprintf(stderr, "%s: %d\n", i8_i32_i64_text,
        //         mlirTypeEqual(i8_i32_i64, i8_i32_i64_ref));

        // const char *i32_i64_s_text = "!llvm.struct<(i32, i64)>";
        // MlirType i32_i64_s = mlirLLVMStructTypeLiteralGet(ctx, 2, i32_i64_arr, false);
        // MlirType i32_i64_s_ref =
        //     mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i32_i64_s_text));
        // // CHECK: !llvm.struct<(i32, i64)>: 1
        // fprintf(stderr, "%s: %d\n", i32_i64_s_text,
        //         mlirTypeEqual(i32_i64_s, i32_i64_s_ref));
    }
}
