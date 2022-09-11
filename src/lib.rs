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

        assert_eq!(module.as_operation().print().as_str().to_str().unwrap(), "");
    }

    #[test]
    fn build_module_with_dialect() {
        let registry = DialectRegistry::new();
        let context = Context::new();
        context.append_dialect_registry(&registry);
        let module = Module::new(Location::unknown(&context));

        assert_eq!(module.as_operation().print().as_str().to_str().unwrap(), "");
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

        let module = Module::new(Location::unknown(&context));

        let r#type = Type::parse(&context, "memref<?xf32>");

        let function = {
            let region = Region::new();
            let block = Block::new(vec![
                (r#type, Location::unknown(&context)),
                (r#type, Location::unknown(&context)),
            ]);

            block.append_operation({
                let mut state = OperationState::new("arith.constant", Location::unknown(&context));

                state.add_results(vec![Type::parse(&context, "index")]);
                state.add_attributes(vec![(
                    Identifier::new(&context, "value"),
                    Attribute::parse(&context, "0 : index"),
                )]);

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

        module
            .as_operation()
            .region(0)
            .first_block()
            .insert_operation(0, function);

        module.as_operation().dump();
        assert!(module.as_operation().verify());

        assert_eq!(module.as_operation().print().as_str().to_str().unwrap(), "");
    }
}
