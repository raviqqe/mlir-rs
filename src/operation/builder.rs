use crate::{
    attribute::Attribute, block::Block, context::Context, identifier::Identifier,
    location::Location, r#type::Type, region::Region, string_ref::StringRef,
    utility::into_raw_array, value::Value,
};
use mlir_sys::{
    mlirNamedAttributeGet, mlirOperationCreate, mlirOperationStateAddAttributes,
    mlirOperationStateAddOperands, mlirOperationStateAddOwnedRegions, mlirOperationStateAddResults,
    mlirOperationStateAddSuccessors, mlirOperationStateEnableResultTypeInference,
    mlirOperationStateGet, MlirOperationState,
};
use std::marker::PhantomData;

use super::Operation;

/// An operation builder.
pub struct Builder<'c> {
    raw: MlirOperationState,
    _context: PhantomData<&'c Context>,
}

impl<'c> Builder<'c> {
    /// Creates an operation builder.
    pub fn new(name: &str, location: Location<'c>) -> Self {
        Self {
            raw: unsafe {
                mlirOperationStateGet(StringRef::from(name).to_raw(), location.to_raw())
            },
            _context: Default::default(),
        }
    }

    /// Adds results.
    pub fn add_results(mut self, results: &[Type<'c>]) -> Self {
        unsafe {
            mlirOperationStateAddResults(
                &mut self.raw,
                results.len() as isize,
                into_raw_array(results.iter().map(|r#type| r#type.to_raw()).collect()),
            )
        }

        self
    }

    /// Adds operands.
    pub fn add_operands(mut self, operands: &[Value]) -> Self {
        unsafe {
            mlirOperationStateAddOperands(
                &mut self.raw,
                operands.len() as isize,
                into_raw_array(operands.iter().map(|value| value.to_raw()).collect()),
            )
        }

        self
    }

    /// Adds regions.
    pub fn add_regions(mut self, regions: Vec<Region>) -> Self {
        unsafe {
            mlirOperationStateAddOwnedRegions(
                &mut self.raw,
                regions.len() as isize,
                into_raw_array(
                    regions
                        .into_iter()
                        .map(|region| region.into_raw())
                        .collect(),
                ),
            )
        }

        self
    }

    /// Adds successor blocks.
    pub fn add_successors(mut self, successors: &[&Block]) -> Self {
        unsafe {
            mlirOperationStateAddSuccessors(
                &mut self.raw,
                successors.len() as isize,
                into_raw_array(successors.iter().map(|block| block.to_raw()).collect()),
            )
        }

        self
    }

    /// Adds attributes.
    pub fn add_attributes(mut self, attributes: &[(Identifier, Attribute<'c>)]) -> Self {
        unsafe {
            mlirOperationStateAddAttributes(
                &mut self.raw,
                attributes.len() as isize,
                into_raw_array(
                    attributes
                        .iter()
                        .map(|(identifier, attribute)| {
                            mlirNamedAttributeGet(identifier.to_raw(), attribute.to_raw())
                        })
                        .collect(),
                ),
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
        context::Context, dialect_registry::DialectRegistry, utility::register_all_dialects,
    };

    #[test]
    fn new() {
        Builder::new("foo", Location::unknown(&Context::new())).build();
    }

    #[test]
    fn add_results() {
        let context = Context::new();

        Builder::new("foo", Location::unknown(&context))
            .add_results(&[Type::parse(&context, "i1").unwrap()])
            .build();
    }

    #[test]
    fn add_regions() {
        let context = Context::new();

        Builder::new("foo", Location::unknown(&context))
            .add_regions(vec![Region::new()])
            .build();
    }

    #[test]
    fn add_successors() {
        let context = Context::new();

        Builder::new("foo", Location::unknown(&context))
            .add_successors(&[&Block::new(&[])])
            .build();
    }

    #[test]
    fn add_attributes() {
        let context = Context::new();

        Builder::new("foo", Location::unknown(&context))
            .add_attributes(&[(
                Identifier::new(&context, "foo"),
                Attribute::parse(&context, "unit").unwrap(),
            )])
            .build();
    }

    #[test]
    fn enable_result_type_inference() {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, location)]);
        let argument = *block.argument(0).unwrap();

        assert_eq!(
            Builder::new("arith.addi", location)
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
