//! `arith` dialect

use crate::{
    ir::{operation, Attribute, Identifier, IntegerAttribute, Location, Operation, Type, Value},
    Context,
};

// spell-checker: disable

/// Creates an `arith.constant` operation.
pub fn constant<'c>(
    context: &'c Context,
    value: Attribute<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    operation::Builder::new("arith.constant", location)
        .add_attributes(&[(Identifier::new(context, "value"), value)])
        .enable_result_type_inference()
        .build()
}

// `arith.cmpf` predicate
pub enum CmpfPredicate {
    False,
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    One,
    Ord,
    Ueq,
    Ugt,
    Uge,
    Ult,
    Ule,
    Une,
    Uno,
    True,
}

/// Creates an `arith.cmpf` operation.
pub fn cmpf<'c>(
    context: &'c Context,
    name: &str,
    predicate: CmpfPredicate,
    lhs: Value,
    rhs: Value,
    location: Location<'c>,
) -> Operation<'c> {
    operation::Builder::new(name, location)
        .add_attributes(&[(
            Identifier::new(context, "predicate"),
            IntegerAttribute::new(predicate as i64, Type::integer(&context, 64)).into(),
        )])
        .add_operands(&[lhs, rhs])
        .enable_result_type_inference()
        .build()
}

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

melior_macro::binary_operations!(
    arith,
    [
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
    ]
);

melior_macro::unary_operations!(arith, [negf, truncf]);

melior_macro::typed_unary_operations!(
    arith,
    [
        bitcast,
        extf,
        extsi,
        extui,
        fptosi,
        fptoui,
        index_cast,
        index_castui,
        sitofp,
        trunci,
        uitofp
    ]
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

    fn create_context() -> Context {
        let context = Context::new();
        load_all_dialects(&context);
        context
    }

    fn compile_operation<'c>(
        context: &'c Context,
        operation: impl Fn(&Block<'c>) -> Operation<'c>,
        block_argument_types: &[Type<'c>],
        function_type: &str,
    ) {
        let location = Location::unknown(context);
        let module = Module::new(location);

        let region = Region::new();
        let block = Block::new(
            &block_argument_types
                .iter()
                .map(|&r#type| (r#type, location))
                .collect::<Vec<_>>(),
        );

        let operation = operation(&block);
        let name = operation.name();
        let name = name.as_string_ref().as_str().unwrap();

        block.append_operation(func::r#return(
            &[block.append_operation(operation).result(0).unwrap().into()],
            location,
        ));

        region.append_block(block);

        let function = func::func(
            context,
            Attribute::parse(context, "\"foo\"").unwrap(),
            Attribute::parse(context, function_type).unwrap(),
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

        compile_operation(
            &context,
            |_| {
                constant(
                    &context,
                    Attribute::parse(&context, "42 : i64").unwrap(),
                    Location::unknown(&context),
                )
            },
            &[Type::integer(&context, 64)],
            "(i64) -> i64",
        );
    }

    #[test]
    fn compile_negf() {
        let context = create_context();

        compile_operation(
            &context,
            |block| {
                negf(
                    block.argument(0).unwrap().into(),
                    Location::unknown(&context),
                )
            },
            &[Type::float64(&context)],
            "(f64) -> f64",
        );
    }

    mod typed_unary {
        use super::*;

        #[test]
        fn compile_bitcast() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    bitcast(
                        block.argument(0).unwrap().into(),
                        Type::float64(&context),
                        Location::unknown(&context),
                    )
                },
                &[Type::integer(&context, 64)],
                "(i64) -> f64",
            );
        }

        #[test]
        fn compile_extf() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    extf(
                        block.argument(0).unwrap().into(),
                        Type::float64(&context),
                        Location::unknown(&context),
                    )
                },
                &[Type::float32(&context)],
                "(f32) -> f64",
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
                        Type::integer(&context, 64),
                        Location::unknown(&context),
                    )
                },
                &[Type::integer(&context, 32)],
                "(i32) -> i64",
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
                        Type::integer(&context, 64),
                        Location::unknown(&context),
                    )
                },
                &[Type::integer(&context, 32)],
                "(i32) -> i64",
            );
        }

        #[test]
        fn compile_fptosi() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    fptosi(
                        block.argument(0).unwrap().into(),
                        Type::integer(&context, 64),
                        Location::unknown(&context),
                    )
                },
                &[Type::float32(&context)],
                "(f32) -> i64",
            );
        }

        #[test]
        fn compile_fptoui() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    fptoui(
                        block.argument(0).unwrap().into(),
                        Type::integer(&context, 64),
                        Location::unknown(&context),
                    )
                },
                &[Type::float32(&context)],
                "(f32) -> i64",
            );
        }

        #[test]
        fn compile_index_cast() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    index_cast(
                        block.argument(0).unwrap().into(),
                        Type::integer(&context, 64),
                        Location::unknown(&context),
                    )
                },
                &[Type::index(&context)],
                "(index) -> i64",
            );
        }

        #[test]
        fn compile_index_castui() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    index_castui(
                        block.argument(0).unwrap().into(),
                        Type::integer(&context, 64),
                        Location::unknown(&context),
                    )
                },
                &[Type::index(&context)],
                "(index) -> i64",
            );
        }

        #[test]
        fn compile_sitofp() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    sitofp(
                        block.argument(0).unwrap().into(),
                        Type::float64(&context),
                        Location::unknown(&context),
                    )
                },
                &[Type::integer(&context, 32)],
                "(i32) -> f64",
            );
        }

        #[test]
        fn compile_trunci() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    trunci(
                        block.argument(0).unwrap().into(),
                        Type::integer(&context, 32),
                        Location::unknown(&context),
                    )
                },
                &[Type::integer(&context, 64)],
                "(i64) -> i32",
            );
        }

        #[test]
        fn compile_uitofp() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    uitofp(
                        block.argument(0).unwrap().into(),
                        Type::float64(&context),
                        Location::unknown(&context),
                    )
                },
                &[Type::integer(&context, 32)],
                "(i32) -> f64",
            );
        }
    }

    #[test]
    fn compile_addi() {
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
                Attribute::parse(&context, "\"foo\"").unwrap(),
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
