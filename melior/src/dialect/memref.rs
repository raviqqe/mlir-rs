//! `memref` dialect.

use crate::{
    ir::{
        attribute::{DenseI32ArrayAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::MemRefType,
        Identifier, Location, Operation, Value,
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
        DenseI32ArrayAttribute::new(
            &context,
            &[dynamic_sizes.len() as i32, symbols.len() as i32],
        )
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
        dialect::func,
        ir::{
            attribute::{StringAttribute, TypeAttribute},
            r#type::FunctionType,
            Block, Module, Region, Type,
        },
        test::create_test_context,
    };

    fn compile_operation(name: &str, context: &Context, build_block: impl Fn(&Block)) {
        let location = Location::unknown(&context);
        let module = Module::new(location);

        let function = {
            let block = Block::new(&[]);

            build_block(&block);
            block.append_operation(func::r#return(&[], location));

            let region = Region::new();
            region.append_block(block);

            func::func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
                region,
                Location::unknown(&context),
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
            let pointer = block.append_operation(alloc(
                &context,
                MemRefType::new(Type::index(&context), &[], None, None),
                &[],
                &[],
                None,
                location,
            ));
            block.append_operation(dealloc(pointer.result(0).unwrap().into(), location));
        })
    }

    #[test]
    fn compile_alloc_and_realloc() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("realloc", &context, |block| {
            let pointer = block.append_operation(alloc(
                &context,
                MemRefType::new(Type::index(&context), &[8], None, None),
                &[],
                &[],
                None,
                location,
            ));
            block.append_operation(realloc(
                &context,
                pointer.result(0).unwrap().into(),
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
}
