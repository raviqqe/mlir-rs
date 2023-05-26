//! `llvm` dialect.

use crate::{
    ir::{
        attribute::DenseI64ArrayAttribute, operation::OperationBuilder, Identifier, Location,
        Operation, Value,
    },
    Context,
};

pub mod r#type;

/// Creates a `llvm.insertvalue` operation.
pub fn insert_value<'c>(
    context: &'c Context,
    container: Value,
    position: DenseI64ArrayAttribute<'c>,
    value: Value,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.insertvalue", location)
        .add_attributes(&[(Identifier::new(context, "position"), position.into())])
        .add_operands(&[container, value])
        .enable_result_type_inference()
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{arith, func},
        ir::{
            attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
            r#type::FunctionType,
            Block, Module, Region,
        },
        test::load_all_dialects,
        Context,
    };

    #[test]
    fn compile_insert_value() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let struct_type = r#type::r#struct(&context, &[], false).into();

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

                block.append_operation(insert_value(&context, operand, "assert message", location));

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
