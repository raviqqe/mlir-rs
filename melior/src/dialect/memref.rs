//! `memref` dialect.

use crate::{
    ir::{
        attribute::{
            DenseI32ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute,
            TypeAttribute,
        },
        operation::OperationBuilder,
        r#type::MemRefType,
        Attribute, Identifier, Location, Operation, Value,
    },
    Context,
};

/// Create a `memref.alloc` operation.
pub fn alloc<'c>(
    context: &'c Context,
    r#type: MemRefType<'c>,
    dynamic_sizes: &[Value],
    symbols: &[Value],
    alignment: Option<IntegerAttribute<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    allocate(
        context,
        "memref.alloc",
        r#type,
        dynamic_sizes,
        symbols,
        alignment,
        location,
    )
}

/// Create a `memref.alloca` operation.
pub fn alloca<'c>(
    context: &'c Context,
    r#type: MemRefType<'c>,
    dynamic_sizes: &[Value],
    symbols: &[Value],
    alignment: Option<IntegerAttribute<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    allocate(
        context,
        "memref.alloca",
        r#type,
        dynamic_sizes,
        symbols,
        alignment,
        location,
    )
}

fn allocate<'c>(
    context: &'c Context,
    name: &str,
    r#type: MemRefType<'c>,
    dynamic_sizes: &[Value],
    symbols: &[Value],
    alignment: Option<IntegerAttribute<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new(name, location);

    builder = builder.add_attributes(&[(
        Identifier::new(context, "operand_segment_sizes"),
        DenseI32ArrayAttribute::new(context, &[dynamic_sizes.len() as i32, symbols.len() as i32])
            .into(),
    )]);
    builder = builder.add_operands(dynamic_sizes).add_operands(symbols);

    if let Some(alignment) = alignment {
        builder =
            builder.add_attributes(&[(Identifier::new(context, "alignment"), alignment.into())]);
    }

    builder.add_results(&[r#type.into()]).build()
}

/// Create a `memref.dealloc` operation.
pub fn dealloc<'c>(value: Value, location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.dealloc", location)
        .add_operands(&[value])
        .build()
}

/// Create a `memref.dim` operation.
pub fn dim<'c>(value: Value, index: Value, location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.dim", location)
        .add_operands(&[value, index])
        .enable_result_type_inference()
        .build()
}

/// Create a `memref.get_global` operation.
pub fn get_global<'c>(context: &'c Context, name: &str, location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.get_global", location)
        .add_attributes(&[(
            Identifier::new(&context, "name"),
            FlatSymbolRefAttribute::new(&context, name).into(),
        )])
        .enable_result_type_inference()
        .build()
}

/// Create a `memref.global` operation.
#[allow(clippy::too_many_arguments)]
pub fn global<'c>(
    context: &'c Context,
    name: &str,
    visibility: Option<&str>,
    r#type: MemRefType<'c>,
    value: Option<Attribute<'c>>,
    constant: bool,
    alignment: Option<IntegerAttribute<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new("memref.global", location).add_attributes(&[
        (
            Identifier::new(context, "sym_name"),
            StringAttribute::new(context, name).into(),
        ),
        (
            Identifier::new(context, "type"),
            TypeAttribute::new(r#type.into()).into(),
        ),
        (
            Identifier::new(context, "initial_value"),
            value.unwrap_or_else(|| Attribute::unit(context)),
        ),
    ]);

    if let Some(visibility) = visibility {
        builder = builder.add_attributes(&[(
            Identifier::new(context, "sym_visibility"),
            StringAttribute::new(context, visibility).into(),
        )]);
    }

    if constant {
        builder = builder.add_attributes(&[(
            Identifier::new(context, "constant"),
            Attribute::unit(context),
        )]);
    }

    if let Some(alignment) = alignment {
        builder =
            builder.add_attributes(&[(Identifier::new(context, "alignment"), alignment.into())]);
    }

    builder.build()
}

/// Create a `memref.load` operation.
pub fn load<'c>(memref: Value, indices: &[Value], location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.load", location)
        .add_operands(&[memref])
        .add_operands(indices)
        .enable_result_type_inference()
        .build()
}

/// Create a `memref.rank` operation.
pub fn rank<'c>(value: Value, location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.rank", location)
        .add_operands(&[value])
        .enable_result_type_inference()
        .build()
}

/// Create a `memref.store` operation.
pub fn store<'c>(
    value: Value,
    memref: Value,
    indices: &[Value],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("memref.store", location)
        .add_operands(&[value, memref])
        .add_operands(indices)
        .build()
}

/// Create a `memref.realloc` operation.
pub fn realloc<'c>(
    context: &'c Context,
    value: Value,
    size: Option<Value>,
    r#type: MemRefType<'c>,
    alignment: Option<IntegerAttribute<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new("memref.realloc", location)
        .add_operands(&[value])
        .add_results(&[r#type.into()]);

    if let Some(size) = size {
        builder = builder.add_operands(&[size]);
    }

    if let Some(alignment) = alignment {
        builder =
            builder.add_attributes(&[(Identifier::new(context, "alignment"), alignment.into())]);
    }

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{func, index},
        ir::{
            attribute::{DenseElementsAttribute, StringAttribute, TypeAttribute},
            r#type::{FunctionType, IntegerType, RankedTensorType},
            Block, Module, Region, Type,
        },
        test::create_test_context,
    };

    fn compile_operation(name: &str, context: &Context, build_block: impl Fn(&Block)) {
        let location = Location::unknown(context);
        let module = Module::new(location);

        let function = {
            let block = Block::new(&[]);

            build_block(&block);
            block.append_operation(func::r#return(&[], location));

            let region = Region::new();
            region.append_block(block);

            func::func(
                context,
                StringAttribute::new(context, "foo"),
                TypeAttribute::new(FunctionType::new(context, &[], &[]).into()),
                region,
                Location::unknown(context),
            )
        };

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(name, module.as_operation());
    }

    #[test]
    fn compile_alloc_and_dealloc() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("alloc", &context, |block| {
            let memref = block.append_operation(alloc(
                &context,
                MemRefType::new(Type::index(&context), &[], None, None),
                &[],
                &[],
                None,
                location,
            ));
            block.append_operation(dealloc(memref.result(0).unwrap().into(), location));
        })
    }

    #[test]
    fn compile_alloc_and_realloc() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("realloc", &context, |block| {
            let memref = block.append_operation(alloc(
                &context,
                MemRefType::new(Type::index(&context), &[8], None, None),
                &[],
                &[],
                None,
                location,
            ));
            block.append_operation(realloc(
                &context,
                memref.result(0).unwrap().into(),
                None,
                MemRefType::new(Type::index(&context), &[42], None, None),
                None,
                location,
            ));
        })
    }

    #[test]
    fn compile_alloca() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("alloca", &context, |block| {
            block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[], None, None),
                &[],
                &[],
                None,
                location,
            ));
        })
    }

    #[test]
    fn compile_dim() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("dim", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[1], None, None),
                &[],
                &[],
                None,
                location,
            ));

            let index = block.append_operation(index::constant(
                &context,
                IntegerAttribute::new(0, Type::index(&context)),
                location,
            ));

            block.append_operation(dim(
                memref.result(0).unwrap().into(),
                index.result(0).unwrap().into(),
                location,
            ));
        })
    }

    #[test]
    fn compile_get_global() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        module.body().append_operation(global(
            &context,
            "foo",
            None,
            MemRefType::new(Type::index(&context), &[], None, None),
            None,
            false,
            None,
            location,
        ));

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);

                block.append_operation(get_global(&context, "foo", location));
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
    fn compile_global() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        module.body().append_operation(global(
            &context,
            "foo",
            None,
            MemRefType::new(Type::index(&context), &[], None, None),
            None,
            false,
            None,
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_global_with_options() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);
        let r#type = IntegerType::new(&context, 64).into();

        module.body().append_operation(global(
            &context,
            "foo",
            Some("private"),
            MemRefType::new(r#type, &[], None, None),
            Some(
                DenseElementsAttribute::new(
                    RankedTensorType::new(&[], r#type, None).into(),
                    &[IntegerAttribute::new(42, r#type).into()],
                )
                .unwrap()
                .into(),
            ),
            true,
            Some(IntegerAttribute::new(
                8,
                IntegerType::new(&context, 64).into(),
            )),
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_load() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("load", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[], None, None),
                &[],
                &[],
                None,
                location,
            ));
            block.append_operation(load(memref.result(0).unwrap().into(), &[], location));
        })
    }

    #[test]
    fn compile_load_with_index() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("load_with_index", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[1], None, None),
                &[],
                &[],
                None,
                location,
            ));

            let index = block.append_operation(index::constant(
                &context,
                IntegerAttribute::new(0, Type::index(&context)),
                location,
            ));

            block.append_operation(load(
                memref.result(0).unwrap().into(),
                &[index.result(0).unwrap().into()],
                location,
            ));
        })
    }

    #[test]
    fn compile_rank() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("rank", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[1], None, None),
                &[],
                &[],
                None,
                location,
            ));
            block.append_operation(rank(memref.result(0).unwrap().into(), location));
        })
    }

    #[test]
    fn compile_store() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("store", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[], None, None),
                &[],
                &[],
                None,
                location,
            ));

            let value = block.append_operation(index::constant(
                &context,
                IntegerAttribute::new(42, Type::index(&context)),
                location,
            ));

            block.append_operation(store(
                value.result(0).unwrap().into(),
                memref.result(0).unwrap().into(),
                &[],
                location,
            ));
        })
    }

    #[test]
    fn compile_store_with_index() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("store_with_index", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[1], None, None),
                &[],
                &[],
                None,
                location,
            ));

            let value = block.append_operation(index::constant(
                &context,
                IntegerAttribute::new(42, Type::index(&context)),
                location,
            ));

            let index = block.append_operation(index::constant(
                &context,
                IntegerAttribute::new(0, Type::index(&context)),
                location,
            ));

            block.append_operation(store(
                value.result(0).unwrap().into(),
                memref.result(0).unwrap().into(),
                &[index.result(0).unwrap().into()],
                location,
            ));
        })
    }
}
