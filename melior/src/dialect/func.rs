//! `func` dialect.

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
        .add_attributes(&[(Identifier::new(&context, "callee"), function.into())])
        .add_operands(arguments)
        .build()
}

/// Create a `func.call_indirect` operation.
pub fn call_indirect<'c>(
    function: Value,
    arguments: &[Value],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("func.call_indirect", location)
        .add_operands(&[function])
        .add_operands(arguments)
        .build()
}

/// Create a `func.constant` operation.
pub fn constant<'c>(
    context: &'c Context,
    function: FlatSymbolRefAttribute<'c>,
    r#type: FunctionType<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("func.constant", location)
        .add_attributes(&[(Identifier::new(&context, "value"), function.into())])
        .add_results(&[r#type.into()])
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
        ir::{attribute::FlatSymbolRefAttribute, r#type::FunctionType, Block, Module, Type},
        test::load_all_dialects,
        Context,
    };

    #[test]
    fn compile_call() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let function = {
            let block = Block::new(&[]);

            block.append_operation(call(
                &context,
                FlatSymbolRefAttribute::new(&context, "foo"),
                &[],
                location,
            ));
            block.append_operation(r#return(&[], location));

            let region = Region::new();
            region.append_block(block);

            func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
                region,
                Location::unknown(&context),
            )
        };

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_call_indirect() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let function = {
            let block = Block::new(&[]);

            let function = block.append_operation(constant(
                &context,
                FlatSymbolRefAttribute::new(&context, "foo"),
                FunctionType::new(&context, &[], &[]),
                location,
            ));
            block.append_operation(call_indirect(
                function.result(0).unwrap().into(),
                &[],
                location,
            ));
            block.append_operation(r#return(&[], location));

            let region = Region::new();
            region.append_block(block);

            func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
                region,
                Location::unknown(&context),
            )
        };

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

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
