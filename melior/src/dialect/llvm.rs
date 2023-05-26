//! `llvm` dialect.

use crate::{
    ir::{
        attribute::{DenseI32ArrayAttribute, DenseI64ArrayAttribute, TypeAttribute},
        operation::OperationBuilder,
        Identifier, Location, Operation, Type, Value,
    },
    Context,
};

pub mod r#type;

// spell-checker: disable

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

/// Creates a `llvm.getelementptr` operation.
pub fn get_element_ptr<'c>(
    context: &'c Context,
    ptr: Value,
    indices: DenseI32ArrayAttribute<'c>,
    element_type: Type<'c>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.getelementptr", location)
        .add_attributes(&[
            (
                Identifier::new(context, "rawConstantIndices"),
                indices.into(),
            ),
            (
                Identifier::new(context, "elem_type"),
                TypeAttribute::new(element_type).into(),
            ),
        ])
        .add_operands(&[ptr])
        .add_results(&[result_type])
        .build()
}

/// Creates a `llvm.getelementptr` operation with dynamic indices.
pub fn get_element_ptr_dynamic<'c, const N: usize>(
    context: &'c Context,
    ptr: Value,
    indices: &[Value<'c>; N],
    element_type: Type<'c>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.getelementptr", location)
        .add_attributes(&[
            (
                Identifier::new(context, "rawConstantIndices"),
                DenseI32ArrayAttribute::new(&context, &[i32::MIN; N]).into(),
            ),
            (
                Identifier::new(context, "elem_type"),
                TypeAttribute::new(element_type).into(),
            ),
        ])
        .add_operands(&[ptr])
        .add_operands(indices)
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
    fn compile_get_element_ptr() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = r#type::opaque_pointer(&context);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[ptr_type], &[]).into()),
            {
                let block = Block::new(&[(ptr_type, location)]);

                block.append_operation(get_element_ptr(
                    &context,
                    block.argument(0).unwrap().into(),
                    DenseI32ArrayAttribute::new(&context, &[42]),
                    integer_type,
                    ptr_type,
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
    fn compile_get_element_ptr_dynamic() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = r#type::opaque_pointer(&context);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[ptr_type], &[]).into()),
            {
                let block = Block::new(&[(ptr_type, location)]);

                let index = block
                    .append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(42, integer_type).into(),
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(get_element_ptr_dynamic(
                    &context,
                    block.argument(0).unwrap().into(),
                    &[index],
                    integer_type,
                    ptr_type,
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
