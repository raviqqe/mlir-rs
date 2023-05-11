//! `memref` dialect.

use crate::{
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::FunctionType,
        Identifier, Location, Operation, Region, Value,
    },
    Context,
};

/// Create a `func.call` operation.
pub fn call<'c>(
    context: &'c Context,
    function: FlatSymbolRefAttribute<'c>,
    arguments: &[Value],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("func.call", location)
        .add_attributes(&[(Identifier::new(context, "callee"), function.into())])
        .add_operands(arguments)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{Block, Module, Type},
        test::load_all_dialects,
    };

    #[test]
    fn compile_function() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let integer_type = Type::index(&context);

        let function = {
            let block = Block::new(&[(integer_type, location)]);

            block.append_operation(r#return(&[block.argument(0).unwrap().into()], location));

            let region = Region::new();
            region.append_block(block);

            func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(
                    FunctionType::new(&context, &[integer_type], &[integer_type]).into(),
                ),
                region,
                Location::unknown(&context),
            )
        };

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
}
