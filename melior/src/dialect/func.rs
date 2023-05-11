//! `func` dialect.

use crate::{
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        Identifier, Location, Operation, Region, Value,
    },
    Context,
};

/// Create a `func.call` operation.
pub fn call<'c>(function: Value, arguments: &[Value], location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("func.return", location)
        .add_operands(operands)
        .build()
}

/// Create a `func.func` operation.
pub fn func<'c>(
    context: &'c Context,
    name: StringAttribute<'c>,
    r#type: TypeAttribute<'c>,
    region: Region,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (Identifier::new(context, "sym_name"), name.into()),
            (Identifier::new(context, "function_type"), r#type.into()),
        ])
        .add_regions(vec![region])
        .build()
}

/// Create a `func.return` operation.
pub fn r#return<'c>(operands: &[Value], location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("func.return", location)
        .add_operands(operands)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{r#type::FunctionType, Block, Module, Type},
        test::load_all_dialects,
        Context,
    };

    #[test]
    fn run_on_function_in_nested_module() {
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
                StringAttribute::new(&context, "add"),
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
