use super::Operation;
use crate::{
    context::Context,
    ir::{Attribute, AttributeLike, Block, Identifier, Location, Region, Type, Value},
    string_ref::StringRef,
};
use mlir_sys::{
    mlirNamedAttributeGet, mlirOperationCreate, mlirOperationStateAddAttributes,
    mlirOperationStateAddOperands, mlirOperationStateAddOwnedRegions, mlirOperationStateAddResults,
    mlirOperationStateAddSuccessors, mlirOperationStateEnableResultTypeInference,
    mlirOperationStateGet, MlirOperationState,
};
use std::{marker::PhantomData, mem::forget};

/// An operation builder.
pub struct OperationBuilder<'c> {
    raw: MlirOperationState,
    _context: PhantomData<&'c Context>,
}

impl<'c> OperationBuilder<'c> {
    /// Creates an operation builder.
    pub fn new(name: &str, location: Location<'c>) -> Self {
        Self {
            raw: unsafe {
                mlirOperationStateGet(StringRef::from(name).to_raw(), location.to_raw())
            },
            _context: Default::default(),
        }
    }

    /// Adds a single result.
    pub fn add_result(mut self, result: &Type<'c>) -> Self {
        unsafe { mlirOperationStateAddResults(&mut self.raw, 1, result as *const _ as *const _) }
        self
    }

    /// Adds results.
    pub fn add_results(mut self, results: &[Type<'c>]) -> Self {
        unsafe {
            mlirOperationStateAddResults(
                &mut self.raw,
                results.len() as isize,
                results as *const _ as *const _,
            )
        }

        self
    }

    /// Adds a single operand.
    pub fn add_operand(mut self, operand: &Value<'c, '_>) -> Self {
        unsafe { mlirOperationStateAddOperands(&mut self.raw, 1, operand as *const _ as *const _) }

        self
    }

    /// Adds operands.
    pub fn add_operands(mut self, operands: &[Value<'c, '_>]) -> Self {
        unsafe {
            mlirOperationStateAddOperands(
                &mut self.raw,
                operands.len() as isize,
                operands as *const _ as *const _,
            )
        }

        self
    }

    /// Adds a single region.
    pub fn add_region(mut self, region: Region<'c>) -> Self {
        unsafe {
            mlirOperationStateAddOwnedRegions(&mut self.raw, 1, &region as *const _ as *const _)
        }
        forget(region);

        self
    }

    /// Adds regions.
    pub fn add_regions(mut self, regions: Vec<Region<'c>>) -> Self {
        unsafe {
            mlirOperationStateAddOwnedRegions(
                &mut self.raw,
                regions.len() as isize,
                regions.leak().as_ptr() as *const _ as *const _,
            )
        }

        self
    }

    /// Adds a single successor block.
    pub fn add_successor(mut self, successor: &Block<'c>) -> Self {
        unsafe {
            mlirOperationStateAddSuccessors(&mut self.raw, 1, &successor as *const _ as *const _)
        }

        self
    }

    /// Adds successor blocks.
    // TODO Fix this to ensure blocks are alive while they are referenced by the
    // operation.
    // TODO Should we accept `BlockRef`?
    pub fn add_successors(mut self, successors: &[&Block<'c>]) -> Self {
        unsafe {
            mlirOperationStateAddSuccessors(
                &mut self.raw,
                successors.len() as isize,
                successors
                    .iter()
                    .map(|block| block.to_raw())
                    .collect::<Vec<_>>()
                    .as_ptr() as *const _,
            )
        }

        self
    }

    /// Adds a single attribute.
    pub fn add_attribute(
        mut self,
        name: &Identifier<'c>,
        attribute: &impl AttributeLike<'c>,
    ) -> Self {
        unsafe {
            mlirOperationStateAddAttributes(
                &mut self.raw,
                1,
                &mlirNamedAttributeGet(name.to_raw(), attribute.to_raw()) as *const _,
            )
        }

        self
    }

    /// Adds attributes.
    // TODO Should we accept `NamedAttribute`?
    pub fn add_attributes(mut self, attributes: &[(Identifier<'c>, Attribute<'c>)]) -> Self {
        unsafe {
            mlirOperationStateAddAttributes(
                &mut self.raw,
                attributes.len() as isize,
                attributes
                    .iter()
                    .map(|(identifier, attribute)| {
                        mlirNamedAttributeGet(identifier.to_raw(), attribute.to_raw())
                    })
                    .collect::<Vec<_>>()
                    .as_ptr() as *const _,
            )
        }

        self
    }

    /// Enables result type inference.
    pub fn enable_result_type_inference(mut self) -> Self {
        unsafe { mlirOperationStateEnableResultTypeInference(&mut self.raw) }

        self
    }

    /// Builds an operation.
    pub fn build(mut self) -> Operation<'c> {
        unsafe { Operation::from_raw(mlirOperationCreate(&mut self.raw)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{Block, ValueLike},
        test::create_test_context,
    };

    #[test]
    fn new() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context)).build();
    }

    #[test]
    fn add_operand() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, location)]);
        let argument = block.argument(0).unwrap().into();

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_operand(&argument)
            .build();
    }

    #[test]
    fn add_operands() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, location)]);
        let argument = block.argument(0).unwrap().into();

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_operands(&[argument])
            .build();
    }

    #[test]
    fn add_result() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_result(&Type::parse(&context, "i1").unwrap())
            .build();
    }

    #[test]
    fn add_results() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_results(&[Type::parse(&context, "i1").unwrap()])
            .build();
    }

    #[test]
    fn add_region() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_region(Region::new())
            .build();
    }

    #[test]
    fn add_regions() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_regions(vec![Region::new()])
            .build();
    }

    #[test]
    fn add_successor() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_successor(&Block::new(&[]))
            .build();
    }

    #[test]
    fn add_successors() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_successors(&[&Block::new(&[])])
            .build();
    }

    #[test]
    fn add_attribute() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_attribute(
                &Identifier::new(&context, "foo"),
                &Attribute::parse(&context, "unit").unwrap(),
            )
            .build();
    }

    #[test]
    fn add_attributes() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_attributes(&[(
                Identifier::new(&context, "foo"),
                Attribute::parse(&context, "unit").unwrap(),
            )])
            .build();
    }

    #[test]
    fn enable_result_type_inference() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, location)]);
        let argument = block.argument(0).unwrap().into();

        assert_eq!(
            OperationBuilder::new("arith.addi", location)
                .add_operands(&[argument, argument])
                .enable_result_type_inference()
                .build()
                .result(0)
                .unwrap()
                .r#type(),
            r#type,
        );
    }
}
