//! `llvm` dialect.

use crate::{
    ir::{
        attribute::{DenseI64ArrayAttribute, StringAttribute},
        operation::OperationBuilder,
        Identifier, Location, Operation, Value,
    },
    Context,
};

pub mod r#type;

/// Creates a `llvm.insertvalue` operation.
pub fn insert_value<'c>(
    context: &'c Context,
    r#struct: Value,
    position: DenseI64ArrayAttribute<'c>,
    value: Value,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.insertvalue", location)
        .add_attributes(&[(
            Identifier::new(context, "msg"),
            StringAttribute::new(context, message).into(),
        )])
        .add_operands(&[argument])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{arith, func},
        ir::{
            attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
            r#type::{FunctionType, IntegerType, Type},
            Block, Module, Region,
        },
        test::load_all_dialects,
        Context,
    };

    #[test]
    fn compile_assert() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let bool_type: Type = IntegerType::new(&context, 1).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                let operand = block
                    .append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(1, bool_type).into(),
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(assert(&context, operand, "assert message", location));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
}
