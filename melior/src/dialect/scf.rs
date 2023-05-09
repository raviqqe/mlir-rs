//! `scf` dialect.

use crate::ir::{operation::Builder, Location, Operation, Region, Type, Value};

/// Creates a `scf.condition` operation.
pub fn condition<'c>(
    condition: Value<'c>,
    values: &[Value<'c>],
    location: Location<'c>,
) -> Operation<'c> {
    Builder::new("scf.condition", location)
        .add_operands(&[condition])
        .add_operands(values)
        .build()
}

/// Creates a `scf.for` operation.
pub fn r#for<'c>(
    start: Value<'c>,
    end: Value<'c>,
    step: Value<'c>,
    region: Region,
    location: Location<'c>,
) -> Operation<'c> {
    Builder::new("scf.for", location)
        .add_operands(&[start, end, step])
        .add_regions(vec![region])
        .build()
}

/// Creates a `scf.while` operation.
pub fn r#while<'c>(
    initial: Value<'c>,
    result_types: &[Type<'c>],
    before_region: Region,
    after_region: Region,
    location: Location<'c>,
) -> Operation<'c> {
    Builder::new("scf.while", location)
        .add_operands(&[initial])
        .add_results(result_types)
        .add_regions(vec![before_region, after_region])
        .build()
}

/// Creates a `scf.yield` operation.
pub fn r#yield<'c>(values: &[Value<'c>], location: Location<'c>) -> Operation<'c> {
    Builder::new("scf.yield", location)
        .add_operands(values)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{arith, func},
        ir::{
            attribute,
            r#type::{self, Type},
            Attribute, Block, Module,
        },
        test::load_all_dialects,
        Context,
    };

    #[test]
    fn compile_for() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        module.body().append_operation(func::func(
            &context,
            Attribute::parse(&context, "\"foo\"").unwrap(),
            Attribute::parse(&context, "() -> ()").unwrap(),
            {
                let block = Block::new(&[]);

                let start = block.append_operation(arith::constant(
                    &context,
                    Attribute::parse(&context, "0 : index").unwrap(),
                    location,
                ));

                let end = block.append_operation(arith::constant(
                    &context,
                    Attribute::parse(&context, "8 : index").unwrap(),
                    location,
                ));

                let step = block.append_operation(arith::constant(
                    &context,
                    Attribute::parse(&context, "1 : index").unwrap(),
                    location,
                ));

                block.append_operation(r#for(
                    start.result(0).unwrap().into(),
                    end.result(0).unwrap().into(),
                    step.result(0).unwrap().into(),
                    {
                        let block = Block::new(&[(Type::index(&context), location)]);
                        block.append_operation(r#yield(&[], location));

                        let region = Region::new();
                        region.append_block(block);
                        region
                    },
                    location,
                ));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_while() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let index_type = Type::index(&context);

        module.body().append_operation(func::func(
            &context,
            Attribute::parse(&context, "\"foo\"").unwrap(),
            Attribute::parse(&context, "() -> ()").unwrap(),
            {
                let block = Block::new(&[]);

                let initial = block.append_operation(arith::constant(
                    &context,
                    attribute::Integer::new(0, index_type).into(),
                    location,
                ));

                block.append_operation(r#while(
                    initial.result(0).unwrap().into(),
                    &[index_type],
                    {
                        let block = Block::new(&[(index_type, location)]);

                        let condition = block.append_operation(arith::constant(
                            &context,
                            attribute::Integer::new(0, r#type::Integer::new(&context, 1).into())
                                .into(),
                            location,
                        ));

                        let result = block.append_operation(arith::constant(
                            &context,
                            attribute::Integer::new(42, Type::index(&context).into()).into(),
                            location,
                        ));

                        block.append_operation(super::condition(
                            condition.result(0).unwrap().into(),
                            &[result.result(0).unwrap().into()],
                            location,
                        ));

                        let region = Region::new();
                        region.append_block(block);
                        region
                    },
                    {
                        let block = Block::new(&[(index_type, location)]);

                        let result = block.append_operation(arith::constant(
                            &context,
                            attribute::Integer::new(42, index_type).into(),
                            location,
                        ));

                        block.append_operation(r#yield(
                            &[result.result(0).unwrap().into()],
                            location,
                        ));

                        let region = Region::new();
                        region.append_block(block);
                        region
                    },
                    location,
                ));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_while_with_different_argument_and_result_types() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let index_type = Type::index(&context);
        let float_type = Type::float64(&context);

        module.body().append_operation(func::func(
            &context,
            Attribute::parse(&context, "\"foo\"").unwrap(),
            Attribute::parse(&context, "() -> ()").unwrap(),
            {
                let block = Block::new(&[]);

                let initial = block.append_operation(arith::constant(
                    &context,
                    attribute::Integer::new(0, index_type).into(),
                    location,
                ));

                block.append_operation(r#while(
                    initial.result(0).unwrap().into(),
                    &[float_type],
                    {
                        let block = Block::new(&[(index_type, location)]);

                        let condition = block.append_operation(arith::constant(
                            &context,
                            attribute::Integer::new(0, r#type::Integer::new(&context, 1).into())
                                .into(),
                            location,
                        ));

                        let result = block.append_operation(arith::constant(
                            &context,
                            attribute::Integer::new(42, Type::index(&context).into()).into(),
                            location,
                        ));

                        block.append_operation(super::condition(
                            condition.result(0).unwrap().into(),
                            &[result.result(0).unwrap().into()],
                            location,
                        ));

                        let region = Region::new();
                        region.append_block(block);
                        region
                    },
                    {
                        let block = Block::new(&[(index_type, location)]);

                        let result = block.append_operation(arith::constant(
                            &context,
                            attribute::Float::new(&context, 42.0, float_type).into(),
                            location,
                        ));

                        block.append_operation(r#yield(
                            &[result.result(0).unwrap().into()],
                            location,
                        ));

                        let region = Region::new();
                        region.append_block(block);
                        region
                    },
                    location,
                ));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
}
