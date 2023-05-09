//! `scf` dialect.

use crate::ir::{operation::Builder, Location, Operation, Region, Value};

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
        ir::{r#type::Type, Attribute, Block, Module},
        test::load_all_dialects,
        Context,
    };

    #[test]
    fn build_sum() {
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
}
