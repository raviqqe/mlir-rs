//! `arith` dialect

use crate::ir::Operation;

/// Creates an `arith.constqnt` operation.
pub fn constant<'c>(
    value: crate::ir::Value,
    location: crate::ir::Location<'c>,
) -> crate::ir::Operation<'c> {
    crate::ir::operation::Builder::new("arith.constant", location)
        .add_operands(&[value])
        .enable_result_type_inference()
        .build()
}

// spell-checker: disable

melior_macro::arith_binary_operators!(
    addf,
    addi,
    addui_extended,
    andi,
    ceildivsi,
    ceildivui,
    divf,
    divsi,
    divui,
    floordivsi,
    maxf,
    maxsi,
    maxui,
    minf,
    minsi,
    minui,
    mulf,
    muli,
    mulsi_extended,
    mului_extended,
    ori,
    remf,
    remsi,
    remui,
    shli,
    shrsi,
    shrui,
    subf,
    subi,
    xori,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::func,
        ir::{Attribute, Block, Location, Module, Region, Type},
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
            let block = Block::new(&[(integer_type, location), (integer_type, location)]);

            let sum = block.append_operation(addi(
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                location,
            ));

            block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], location));

            region.append_block(block);

            func::func(
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
