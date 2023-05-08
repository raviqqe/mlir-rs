//! `func` dialect

use crate::{
    ir::{operation::Builder, Attribute, Identifier, Location, Operation, Region, Value},
    Context,
};

/// Create a `func.func` operation.
pub fn func<'c>(
    context: &'c Context,
    name: Attribute<'c>,
    r#type: Attribute<'c>,
    region: Region,
    location: Location<'c>,
) -> Operation<'c> {
    Builder::new("func.func", location)
        .add_attributes(&[
            (Identifier::new(context, "sym_name"), name),
            (Identifier::new(context, "function_type"), r#type),
        ])
        .add_regions(vec![region])
        .build()
}

/// Create a `func.return` operation.
pub fn r#return<'c>(operands: &[Value], location: Location<'c>) -> Operation<'c> {
    Builder::new("func.return", location)
        .add_operands(operands)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{Attribute, Block, Module, Type},
        test::load_all_dialects,
        Context,
    };

    #[test]
    fn run_on_function_in_nested_module() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let integer_type = Type::integer(&context, 64);

        let function = {
            let region = Region::new();
            let block = Block::new(&[(integer_type, location)]);

            block.append_operation(r#return(&[block.argument(0).unwrap().into()], location));

            region.append_block(block);

            func(
                &context,
                Attribute::parse(&context, "\"add\"").unwrap(),
                Attribute::parse(&context, "(i64) -> i64").unwrap(),
                region,
                Location::unknown(&context),
            )
        };

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
}
