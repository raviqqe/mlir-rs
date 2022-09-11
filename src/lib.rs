pub mod attribute;
pub mod block;
pub mod context;
pub mod dialect;
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

            let zero = function_block.append_operation({
                let mut state = OperationState::new("arith.constant", location);

                state.add_results(vec![index_type]).add_attributes(vec![(
                    Identifier::new(&context, "value"),
                    Attribute::parse(&context, "0 : index"),
                )]);

                Operation::new(state)
            });

            let dim = function_block.append_operation({
                let mut state = OperationState::new("memref.dim", location);

                state
                    .add_operands(vec![function_block.argument(0), zero.result(0)])
                    .add_results(vec![index_type]);

                Operation::new(state)
            });

            let loop_block = Block::new(vec![]);
            loop_block.add_argument(index_type, location);

            let one = function_block.append_operation({
                let mut state = OperationState::new("arith.constant", location);

                state.add_results(vec![index_type]).add_attributes(vec![(
                    Identifier::new(&context, "value"),
                    Attribute::parse(&context, "1 : index"),
                )]);

                Operation::new(state)
            });

            {
                let f32_type = Type::parse(&context, "f32");

                let lhs = loop_block.append_operation({
                    let mut state = OperationState::new("memref.load", location);

                    state.add_operands(vec![function_block.argument(0), loop_block.argument(0)]);
                    state.add_results(vec![f32_type]);

                    Operation::new(state)
                });

                let rhs = loop_block.append_operation({
                    let mut state = OperationState::new("memref.load", location);

                    state.add_operands(vec![function_block.argument(1), loop_block.argument(0)]);
                    state.add_results(vec![f32_type]);

                    Operation::new(state)
                });

                let add = loop_block.append_operation({
                    let mut state = OperationState::new("arith.addf", location);

                    state.add_operands(vec![lhs.result(0), rhs.result(0)]);
                    state.add_results(vec![f32_type]);

                    Operation::new(state)
                });

                loop_block.append_operation({
                    let mut state = OperationState::new("memref.store", location);

                    state.add_operands(vec![
                        add.result(0),
                        function_block.argument(0),
                        loop_block.argument(0),
                    ]);

                    Operation::new(state)
                });

                loop_block
                    .append_operation(Operation::new(OperationState::new("scf.yield", location)));
            }

            function_block.append_operation(Operation::new({
                let mut state = OperationState::new("scf.for", location);

                state.add_operands(vec![zero.result(0), dim.result(0), one.result(0)]);
                let loop_region = Region::new();
                loop_region.append_block(loop_block);
                state.add_regions(vec![loop_region]);

                state
            }));

            function_block.append_operation(Operation::new(OperationState::new(
                "func.return",
                Location::unknown(&context),
            )));

            function_region.append_block(function_block);

            let mut state = OperationState::new("func.func", Location::unknown(&context));

            state
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
                .add_regions(vec![function_region]);

            Operation::new(state)
        };

        module.body_mut().insert_operation(0, function);

        module.as_operation().dump();

        assert!(module.as_operation().verify());
        // TODO Fix this. Somehow, MLIR inserts null characters in the middle of string refs.
        // assert_eq!(module.as_operation().print(), "");
    }
}
