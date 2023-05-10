//! `index` dialect.

use crate::{
    ir::{
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Attribute,
        Identifier, Location, Operation, Value,
    },
    Context,
};

// spell-checker: disable

/// Creates an `index.constant` operation.
pub fn constant<'c>(
    context: &'c Context,
    value: Attribute<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("index.constant", location)
        .add_attributes(&[(Identifier::new(context, "value"), value)])
        .enable_result_type_inference()
        .build()
}

/// `index.cmp` predicate
pub enum CmpiPredicate {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
}

/// Creates an `index.cmp` operation.
pub fn cmp<'c>(
    context: &'c Context,
    predicate: CmpiPredicate,
    lhs: Value,
    rhs: Value,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("index.cmp", location)
        .add_attributes(&[(
            Identifier::new(context, "predicate"),
            IntegerAttribute::new(predicate as i64, IntegerType::new(context, 64).into()).into(),
        )])
        .add_operands(&[lhs, rhs])
        .enable_result_type_inference()
        .build()
}

melior_macro::binary_operations!(
    index,
    [
        add, and, ceildivs, ceildivu, divs, divu, floordivs, maxs, maxu, mins, minu, mul, or, rems,
        remu, shl, shrs, shrui, sub, xor,
    ]
);

melior_macro::typed_unary_operations!(index, [casts, castu]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::func,
        ir::{
            attribute::{StringAttribute, TypeAttribute},
            r#type::FunctionType,
            Attribute, Block, Location, Module, Region, Type,
        },
        test::load_all_dialects,
        Context,
    };

    fn create_context() -> Context {
        let context = Context::new();
        load_all_dialects(&context);
        context
    }

    fn compile_operation<'c>(
        context: &'c Context,
        operation: impl Fn(&Block<'c>) -> Operation<'c>,
        function_type: FunctionType<'c>,
    ) {
        let location = Location::unknown(context);
        let module = Module::new(location);

        let block = Block::new(
            &(0..function_type.input_count())
                .map(|index| (function_type.input(index).unwrap(), location))
                .collect::<Vec<_>>(),
        );

        let operation = operation(&block);
        let name = operation.name();
        let name = name.as_string_ref().as_str().unwrap();

        block.append_operation(func::r#return(
            &[block.append_operation(operation).result(0).unwrap().into()],
            location,
        ));

        let region = Region::new();
        region.append_block(block);

        let function = func::func(
            context,
            StringAttribute::new(context, "foo"),
            TypeAttribute::new(function_type.into()),
            region,
            Location::unknown(context),
        );

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(name, module.as_operation());
    }

    #[test]
    fn compile_constant() {
        let context = create_context();
        let integer_type = IntegerType::new(&context, 64).into();

        compile_operation(
            &context,
            |_| {
                constant(
                    &context,
                    Attribute::parse(&context, "42 : i64").unwrap(),
                    Location::unknown(&context),
                )
            },
            FunctionType::new(&context, &[integer_type], &[integer_type]),
        );
    }

    #[test]
    fn compile_cmp() {
        let context = create_context();
        let integer_type = IntegerType::new(&context, 64).into();

        compile_operation(
            &context,
            |block| {
                cmp(
                    &context,
                    CmpiPredicate::Eq,
                    block.argument(0).unwrap().into(),
                    block.argument(1).unwrap().into(),
                    Location::unknown(&context),
                )
            },
            FunctionType::new(
                &context,
                &[integer_type, integer_type],
                &[IntegerType::new(&context, 1).into()],
            ),
        );
    }

    mod typed_unary {
        use super::*;

        #[test]
        fn compile_casts() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    casts(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&context, 64).into(),
                        Location::unknown(&context),
                    )
                },
                FunctionType::new(
                    &context,
                    &[Type::float32(&context)],
                    &[IntegerType::new(&context, 64).into()],
                ),
            );
        }

        #[test]
        fn compile_extsi() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    extsi(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&context, 64).into(),
                        Location::unknown(&context),
                    )
                },
                FunctionType::new(
                    &context,
                    &[IntegerType::new(&context, 32).into()],
                    &[IntegerType::new(&context, 64).into()],
                ),
            );
        }

        #[test]
        fn compile_extui() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    extui(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&context, 64).into(),
                        Location::unknown(&context),
                    )
                },
                FunctionType::new(
                    &context,
                    &[IntegerType::new(&context, 32).into()],
                    &[IntegerType::new(&context, 64).into()],
                ),
            );
        }
    }

    #[test]
    fn compile_add() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let integer_type = Type::index(&context);

        let function = {
            let block = Block::new(&[(integer_type, location), (integer_type, location)]);

            let sum = block.append_operation(add(
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                location,
            ));

            block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], location));

            let region = Region::new();
            region.append_block(block);

            func::func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(
                    FunctionType::new(&context, &[integer_type, integer_type], &[integer_type])
                        .into(),
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
