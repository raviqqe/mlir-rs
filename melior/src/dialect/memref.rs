//! `memref` dialect.

use crate::{
    ir::{
        attribute::{DenseI32ArrayAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::MemRefType,
        Identifier, Location, Operation,
    },
    Context,
};

/// Create a `memref.alloc` operation.
pub fn alloc<'c>(
    context: &'c Context,
    r#type: MemRefType<'c>,
    alignment: Option<IntegerAttribute<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new("memref.alloc", location);

    builder = builder.add_attributes(&[(
        Identifier::new(context, "operand_segment_sizes"),
        DenseI32ArrayAttribute::new(&context, &[0, 0]).into(),
    )]);

    if let Some(alignment) = alignment {
        builder =
            builder.add_attributes(&[(Identifier::new(context, "alignment"), alignment.into())]);
    }

    builder.add_results(&[r#type.into()]).build()
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

    #[test]
    fn compile_alloc() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let function = {
            let block = Block::new(&[]);

            block.append_operation(alloc(
                &context,
                MemRefType::new(Type::index(&context), &[], None, None),
                None,
                location,
            ));
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
        insta::assert_display_snapshot!(module.as_operation());
    }
}
