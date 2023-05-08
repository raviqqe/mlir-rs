use crate::{
    ir::{operation::Builder, Attribute, Identifier, Location, Operation, Region, Value},
    Context,
};

pub fn func<'c>(
    context: &'c Context,
    name: Attribute<'c>,
    r#type: Attribute<'c>,
    region: Region,
    location: Location<'c>,
) -> Operation<'c> {
    Builder::new("func.func", location)
        .add_attributes(&[
            (Identifier::new(&context, "sym_name"), name),
            (Identifier::new(&context, "function_type"), r#type),
        ])
        .add_regions(vec![region])
        .build()
}

pub fn r#return<'c>(operands: &[Value], location: Location<'c>) -> Operation<'c> {
    Builder::new("func.return", location)
        .add_operands(operands)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{arith::addi, Registry},
        ir::{Attribute, Block, Module, Type},
        utility::register_all_dialects,
        Context,
    };

    fn load_all_dialects(context: &Context) {
        let registry = Registry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
    }

    #[test]
    fn run_on_function_in_nested_module() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let integer_type = Type::integer(&context, 64);

        let function = {
            let region = Region::new();
            let block = Block::new(&[(integer_type, location), (integer_type, location)]);

            let sum = block.append_operation(addi(
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                location,
            ));

            block.append_operation(r#return(&[sum.result(0).unwrap().into()], location));

            region.append_block(block);

            func(
                &context,
                Attribute::parse(&context, "\"add\"").unwrap(),
                Attribute::parse(&context, "(i64, i64) -> i64").unwrap(),
                region,
                Location::unknown(&context),
            )
        };

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
}
