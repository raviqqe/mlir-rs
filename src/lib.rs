pub mod attribute;
pub mod block;
pub mod context;
pub mod dialect;
pub mod dialect_handle;
pub mod dialect_registry;
pub mod identifier;
pub mod location;
pub mod module;
pub mod operation;
pub mod operation_state;
pub mod region;
pub mod string_ref;
pub mod r#type;
pub mod utility;
pub mod value;

#[cfg(test)]
mod tests {
    use mlir_sys::mlirGetDialectHandle__func__;

    use crate::{
        attribute::Attribute, block::Block, context::Context, dialect_registry::DialectRegistry,
        identifier::Identifier, location::Location, module::Module, operation::Operation,
        operation_state::OperationState, r#type::Type, region::Region,
    };

    #[test]
    fn build_module() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));

        assert!(module.as_operation().verify());
        assert_eq!(module.as_operation().print(), "module{}");
    }

    #[test]
    fn build_module_with_dialect() {
        let registry = DialectRegistry::new();
        let context = Context::new();
        context.append_dialect_registry(&registry);
        let module = Module::new(Location::unknown(&context));

        assert!(module.as_operation().verify());
        assert_eq!(module.as_operation().print(), "module{}");
    }

    // This test is copied directly from a `makeAndDumpAdd` function in:
    // https://github.com/llvm/llvm-project/blob/llvmorg-15.0.0/mlir/test/CAPI/ir.c
    #[test]
    fn build_add() {
        let registry = DialectRegistry::new();
        registry.register_all_dialects();

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("func");
        context.get_or_load_dialect("memref");
        context.get_or_load_dialect("shape");
        context.get_or_load_dialect("scf");

        let location = Location::unknown(&context);
        let mut module = Module::new(location);

        let r#type = Type::parse(&context, "memref<?xf32>");

        let function = {
            let function_region = Region::new();
            let function_block = Block::new(vec![(r#type, location), (r#type, location)]);
            let index_type = Type::parse(&context, "index");

            let zero = function_block.append_operation(Operation::new(
                OperationState::new("arith.constant", location)
                    .add_results(vec![index_type])
                    .add_attributes(vec![(
                        Identifier::new(&context, "value"),
                        Attribute::parse(&context, "0 : index"),
                    )]),
            ));

            let dim = function_block.append_operation(Operation::new(
                OperationState::new("memref.dim", location)
                    .add_operands(vec![
                        function_block.argument(0).unwrap(),
                        zero.result(0).unwrap(),
                    ])
                    .add_results(vec![index_type]),
            ));

            let loop_block = Block::new(vec![]);
            loop_block.add_argument(index_type, location);

            let one = function_block.append_operation(Operation::new(
                OperationState::new("arith.constant", location)
                    .add_results(vec![index_type])
                    .add_attributes(vec![(
                        Identifier::new(&context, "value"),
                        Attribute::parse(&context, "1 : index"),
                    )]),
            ));

            {
                let f32_type = Type::parse(&context, "f32");

                let lhs = loop_block.append_operation(Operation::new(
                    OperationState::new("memref.load", location)
                        .add_operands(vec![
                            function_block.argument(0).unwrap(),
                            loop_block.argument(0).unwrap(),
                        ])
                        .add_results(vec![f32_type]),
                ));

                let rhs = loop_block.append_operation(Operation::new(
                    OperationState::new("memref.load", location)
                        .add_operands(vec![
                            function_block.argument(1).unwrap(),
                            loop_block.argument(0).unwrap(),
                        ])
                        .add_results(vec![f32_type]),
                ));

                let add = loop_block.append_operation(Operation::new(
                    OperationState::new("arith.addf", location)
                        .add_operands(vec![lhs.result(0).unwrap(), rhs.result(0).unwrap()])
                        .add_results(vec![f32_type]),
                ));

                loop_block.append_operation(Operation::new(
                    OperationState::new("memref.store", location).add_operands(vec![
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
                    .add_operands(vec![
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
                    .add_attributes(vec![
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

        module.body_mut().insert_operation(0, function);

        assert!(module.as_operation().verify());
        // TODO Fix this. Somehow, MLIR inserts null characters in the middle of string refs.
        // assert_eq!(module.as_operation().print(), "");
    }

    fn dialect_registry() {
        let registry = DialectRegistry::new();

        let handle = DialectHandle::func();
        handle.insert_dialect(registry);

        let context = Context::new();
        assert_eq!(context.registered_dialect_count(), 0);

        context.append_dialect_registry(&registry);

        assert_eq!(context.registered_dialect_count(), 1);
    }
}
