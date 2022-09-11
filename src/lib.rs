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
            let region = Region::new();
            let mut block = Block::new(vec![(r#type, location), (r#type, location)]);
            let index_type = Type::parse(&context, "index");

            block.append_operation({
                let mut state = OperationState::new("arith.constant", location);

                state.add_results(vec![index_type]);
                state.add_attributes(vec![(
                    Identifier::new(&context, "value"),
                    Attribute::parse(&context, "0 : index"),
                )]);

                Operation::new(state)
            });

            block.append_operation({
                let mut state = OperationState::new("memref.dim", location);

                state.add_operands(vec![
                    block.argument(0),
                    block.first_operation().unwrap().result(0),
                ]);
                state.add_results(vec![index_type]);

                Operation::new(state)
            });

            // TODO

            block.append_operation(Operation::new(OperationState::new(
                "func.return",
                Location::unknown(&context),
            )));

            region.append_block(block);

            let mut state = OperationState::new("func.func", Location::unknown(&context));

            state.add_attributes(vec![
                (
                    Identifier::new(&context, "function_type"),
                    Attribute::parse(&context, "(memref<?xf32>, memref<?xf32>) -> ()"),
                ),
                (
                    Identifier::new(&context, "sym_name"),
                    Attribute::parse(&context, "\"add\""),
                ),
            ]);
            state.add_regions(vec![region]);

            Operation::new(state)
        };

        module.body_mut().insert_operation(0, function);

        module.as_operation().dump();

        assert!(module.as_operation().verify());
        // TODO Fix this. Somehow, MLIR inserts null characters in the middle of string refs.
        // assert_eq!(module.as_operation().print(), "");
    }
}
