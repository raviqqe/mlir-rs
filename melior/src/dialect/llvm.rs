//! `llvm` dialect.

use crate::{
    ir::{
        attribute::{
            ArrayAttribute, DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute,
            StringAttribute, TypeAttribute,
        },
        operation::OperationBuilder,
        Attribute, Identifier, Location, Operation, Region, Type, Value,
    },
    Context,
};

pub mod attributes;
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
                DenseI32ArrayAttribute::new(context, &[i32::MIN; N]).into(),
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

/// Creates a `llvm.mlir.undef` operation.
pub fn undef<'c>(result_type: Type<'c>, location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("llvm.mlir.undef", location)
        .add_results(&[result_type])
        .build()
}

/// Creates a `llvm.mlir.poison` operation.
pub fn poison<'c>(result_type: Type<'c>, location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("llvm.mlir.poison", location)
        .add_results(&[result_type])
        .build()
}

#[derive(Debug, Default)]
pub struct LoadStoreOptions<'c> {
    pub align: Option<IntegerAttribute<'c>>,
    pub volatile: bool,
    pub nontemporal: bool,
    pub access_groups: Option<ArrayAttribute<'c>>,
    pub alias_scopes: Option<ArrayAttribute<'c>>,
    pub noalias_scopes: Option<ArrayAttribute<'c>>,
    pub tbaa: Option<ArrayAttribute<'c>>,
}

impl<'c> LoadStoreOptions<'c> {
    fn into_attributes(self, context: &'c Context) -> Vec<(Identifier<'c>, Attribute<'c>)> {
        let mut attributes = Vec::with_capacity(7);

        if let Some(align) = self.align {
            attributes.push((Identifier::new(context, "alignment"), align.into()));
        }

        if self.volatile {
            attributes.push((
                Identifier::new(context, "volatile_"),
                Attribute::unit(context),
            ));
        }

        if self.nontemporal {
            attributes.push((
                Identifier::new(context, "nontemporal"),
                Attribute::unit(context),
            ));
        }

        if let Some(alias_scopes) = self.alias_scopes {
            attributes.push((
                Identifier::new(context, "alias_scopes"),
                alias_scopes.into(),
            ));
        }

        if let Some(noalias_scopes) = self.noalias_scopes {
            attributes.push((
                Identifier::new(context, "noalias_scopes"),
                noalias_scopes.into(),
            ));
        }

        if let Some(tbaa) = self.tbaa {
            attributes.push((Identifier::new(context, "tbaa"), tbaa.into()));
        }

        attributes
    }
}

/// Creates a `llvm.store` operation.
pub fn store<'c>(
    context: &'c Context,
    value: Value,
    addr: Value,
    location: Location<'c>,
    extra_options: LoadStoreOptions<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.store", location)
        .add_operands(&[value, addr])
        .add_attributes(&extra_options.into_attributes(context))
        .build()
}

/// Creates a `llvm.load` operation.
pub fn load<'c>(
    context: &'c Context,
    addr: Value,
    r#type: Type<'c>,
    location: Location<'c>,
    extra_options: LoadStoreOptions<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.load", location)
        .add_operands(&[addr])
        .add_attributes(&extra_options.into_attributes(context))
        .add_results(&[r#type])
        .build()
}

/// Create a `llvm.func` operation.
pub fn func<'c>(
    context: &'c Context,
    name: StringAttribute<'c>,
    r#type: TypeAttribute<'c>,
    region: Region,
    attributes: &[(Identifier<'c>, Attribute<'c>)],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.func", location)
        .add_attributes(&[
            (Identifier::new(context, "sym_name"), name.into()),
            (Identifier::new(context, "function_type"), r#type.into()),
        ])
        .add_attributes(attributes)
        .add_regions(vec![region])
        .build()
}

// Creates a `llvm.return` operation.
pub fn r#return<'c>(value: Option<Value>, location: Location<'c>) -> Operation<'c> {
    let mut builder = OperationBuilder::new("llvm.return", location);

    if let Some(value) = value {
        builder = builder.add_operands(&[value]);
    }

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{
            arith, func,
            llvm::{
                attributes::{linkage, Linkage},
                r#type::{function, opaque_pointer},
            },
        },
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

    #[test]
    fn compile_poison() {
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

                block.append_operation(poison(struct_type, location));

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
    fn compile_store() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = r#type::opaque_pointer(&context);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[ptr_type, integer_type], &[]).into()),
            {
                let block = Block::new(&[(ptr_type, location), (integer_type, location)]);

                block.append_operation(store(
                    &context,
                    block.argument(1).unwrap().into(),
                    block.argument(0).unwrap().into(),
                    location,
                    Default::default(),
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
    fn compile_load() {
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

                block.append_operation(load(
                    &context,
                    block.argument(0).unwrap().into(),
                    integer_type,
                    location,
                    Default::default(),
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
    fn compile_store_extra() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = r#type::opaque_pointer(&context);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[ptr_type, integer_type], &[]).into()),
            {
                let block = Block::new(&[(ptr_type, location), (integer_type, location)]);

                block.append_operation(store(
                    &context,
                    block.argument(1).unwrap().into(),
                    block.argument(0).unwrap().into(),
                    location,
                    LoadStoreOptions {
                        align: Some(IntegerAttribute::new(4, integer_type)),
                        volatile: true,
                        nontemporal: true,
                        access_groups: None,
                        alias_scopes: None,
                        noalias_scopes: None,
                        tbaa: None,
                    },
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
    fn compile_func() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 32).into();

        module.body().append_operation(func(
            &context,
            StringAttribute::new(&context, "printf"),
            TypeAttribute::new(function(integer_type, &[opaque_pointer(&context)], true)),
            Region::new(),
            &[(
                Identifier::new(&context, "linkage"),
                linkage(&context, Linkage::External),
            )],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_return() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let struct_type = r#type::r#struct(&context, &[integer_type], false);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo_none"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);

                block.append_operation(r#return(None, location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[struct_type], &[struct_type]).into()),
            {
                let block = Block::new(&[(struct_type, location)]);

                block.append_operation(r#return(Some(block.argument(0).unwrap().into()), location));

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
