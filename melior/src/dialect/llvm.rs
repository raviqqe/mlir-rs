//! `llvm` dialect.

use crate::{
    ir::{
        attribute::DenseI64ArrayAttribute, operation::OperationBuilder, Identifier, Location,
        Operation, Type, Value,
    },
    Context,
};

pub mod r#type;

/// Creates a `llvm.extractvalue` operation.
pub fn extract_value<'c>(
    context: &'c Context,
    container: Value,
    position: DenseI64ArrayAttribute<'c>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.extractvalue", location)
        .add_attributes(&[(Identifier::new(context, "position"), position.into())])
        .add_operands(&[container])
        .add_results(&[result_type])
        .build()
}

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

/// Creates a `llvm.undef` operation.
pub fn undef<'c>(result_type: Type<'c>, location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("llvm.undef", location)
        .add_results(&[result_type])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{arith, func},
        ir::{
            attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
            r#type::{FunctionType, IntegerType},
            Block, Module, Region,
        },
        pass::{self, PassManager},
        test::create_test_context,
    };

    fn convert_module<'c>(context: &'c Context, module: &mut Module<'c>) {
        let pass_manager = PassManager::new(context);

        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_index_to_llvm_pass());
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_mem_ref_to_llvm());

        assert_eq!(pass_manager.run(module), Ok(()));
        assert!(module.as_operation().verify());
    }

    #[test]
    fn compile_extract_value() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let struct_type = r#type::r#struct(&context, &[integer_type], false);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[struct_type], &[]).into()),
            {
                let block = Block::new(&[(struct_type, location)]);

                block.append_operation(extract_value(
                    &context,
                    block.argument(0).unwrap().into(),
                    DenseI64ArrayAttribute::new(&context, &[0]),
                    integer_type,
                    location,
                ));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_insert_value() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let struct_type = r#type::r#struct(&context, &[integer_type], false);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[struct_type], &[]).into()),
            {
                let block = Block::new(&[(struct_type, location)]);
                let value = block
                    .append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(42, integer_type).into(),
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(insert_value(
                    &context,
                    block.argument(0).unwrap().into(),
                    DenseI64ArrayAttribute::new(&context, &[0]),
                    value,
                    location,
                ));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_undefined() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let struct_type =
            r#type::r#struct(&context, &[IntegerType::new(&context, 64).into()], false);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[struct_type], &[]).into()),
            {
                let block = Block::new(&[(struct_type, location)]);

                block.append_operation(undef(struct_type, location));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
}
