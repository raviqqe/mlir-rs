//! `scf` dialect.

use std::dbg;

use crate::{
    ir::{
        attribute::{
            DenseElementsAttribute, DenseI32ArrayAttribute, IntegerAttribute, StringAttribute,
        },
        operation::OperationBuilder,
        r#type::RankedTensorType,
        Attribute, Block, Identifier, Location, Operation, Type, Value,
    },
    Context, Error,
};

/// Creates a `cf.assert` operation.
pub fn assert<'c>(
    context: &'c Context,
    arg: Value<'c>,
    msg: &str,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("cf.assert", location)
        .add_attributes(&[(
            Identifier::new(context, "msg"),
            StringAttribute::new(context, msg).into(),
        )])
        .add_operands(&[arg])
        .build()
}

/// Creates a `cf.br` operation.
pub fn br<'c>(
    successor: &Block<'c>,
    dest_operands: &[Value<'c>],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("cf.br", location)
        .add_operands(dest_operands)
        .add_successors(&[successor])
        .build()
}

/// Creates a `cf.cond_br` operation.
pub fn cond_br<'c>(
    context: &'c Context,
    condition: Value<'c>,
    true_successor: &Block<'c>,
    false_successor: &Block<'c>,
    true_successor_operands: &[Value],
    false_successor_operands: &[Value],
    location: Location<'c>,
) -> Operation<'c> {
    let mut operands = vec![condition];
    operands.extend(true_successor_operands);
    operands.extend(false_successor_operands);

    OperationBuilder::new("cf.cond_br", location)
        .add_attributes(&[(
            Identifier::new(context, "operand_segment_sizes"),
            DenseI32ArrayAttribute::new(
                context,
                &[
                    1,
                    true_successor.argument_count() as i32,
                    false_successor.argument_count() as i32,
                ],
            )
            .into(),
        )])
        .add_operands(&operands)
        .add_successors(&[true_successor, false_successor])
        .build()
}

/// Creates a `cf.switch` operation.
pub fn switch<'c>(
    context: &'c Context,
    case_values: &[i64],
    flag: Value<'c>,
    flag_type: Type<'c>,
    default_destination: (&Block<'c>, &[Value]),
    case_destinations: &[(&Block<'c>, &[Value])],
    location: Location<'c>,
) -> Result<Operation<'c>, Error> {
    let case_segment_sizes: Vec<i32> = case_destinations.iter().map(|x| x.1.len() as i32).collect();

    let default_op_segments = default_destination.1.len() as i32;
    let case_op_segments: i32 = case_destinations.iter().map(|x| x.1.len() as i32).sum();

    let (dests, operands): (Vec<_>, Vec<_>) = std::iter::once(&default_destination)
        .chain(case_destinations.iter())
        .cloned()
        .unzip();

    let attr_case_values: Vec<Attribute> = case_values
        .iter()
        .map(|x| IntegerAttribute::new(*x, flag_type).into())
        .collect();

    let operands: Vec<Value> = std::iter::once([flag].as_slice())
        .chain(operands.into_iter())
        .flatten()
        .cloned()
        .collect();

    Ok(OperationBuilder::new("cf.switch", location)
        .add_attributes(&[
            (
                Identifier::new(context, "case_values"),
                DenseElementsAttribute::new(
                    RankedTensorType::new(&[case_values.len() as u64], flag_type, None).into(),
                    &attr_case_values,
                )?
                .into(),
            ),
            (
                Identifier::new(context, "case_operand_segments"),
                DenseI32ArrayAttribute::new(context, &case_segment_sizes).into(),
            ),
            (
                Identifier::new(context, "operand_segment_sizes"),
                DenseI32ArrayAttribute::new(context, &[1, default_op_segments, case_op_segments])
                    .into(),
            ),
        ])
        .add_operands(&dbg!(operands))
        .add_successors(&dests)
        .build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{
            arith::{self, CmpiPredicate},
            func, index,
        },
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
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_br() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let index_type = Type::index(&context);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                let dest_block = Block::new(&[(index_type, location)]);
                let operand = block
                    .append_operation(index::constant(
                        &context,
                        IntegerAttribute::new(1, index_type),
                        location,
                    ))
                    .result(0)
                    .unwrap();

                block.append_operation(br(&dest_block, &[operand.into()], location));

                dest_block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region.append_block(dest_block);
                region
            },
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_cond_br() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let index_type = Type::index(&context);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                let true_block = Block::new(&[(index_type, location)]);
                let false_block = Block::new(&[(index_type, location)]);

                let operand = block
                    .append_operation(index::constant(
                        &context,
                        IntegerAttribute::new(1, index_type),
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                let condition = block
                    .append_operation(index::cmp(
                        &context,
                        CmpiPredicate::Eq,
                        operand,
                        operand,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(cond_br(
                    &context,
                    condition,
                    &true_block,
                    &false_block,
                    &[operand],
                    &[operand],
                    location,
                ));

                true_block.append_operation(func::r#return(&[], location));
                false_block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region.append_block(true_block);
                region.append_block(false_block);
                region
            },
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_switch() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let i32_type: Type = IntegerType::new(&context, 32).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                let default_block = Block::new(&[(i32_type, location)]);
                let first_block = Block::new(&[(i32_type, location)]);
                let second_block = Block::new(&[(i32_type, location)]);

                let operand = block
                    .append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(1, i32_type).into(),
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(
                    switch(
                        &context,
                        &[0, 1],
                        operand,
                        i32_type,
                        (&default_block, &[operand]),
                        &[(&first_block, &[operand]), (&second_block, &[operand])],
                        location,
                    )
                    .unwrap(),
                );

                default_block.append_operation(func::r#return(&[], location));
                first_block.append_operation(func::r#return(&[], location));
                second_block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region.append_block(default_block);
                region.append_block(first_block);
                region.append_block(second_block);
                region
            },
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
}
