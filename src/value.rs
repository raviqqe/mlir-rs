use crate::r#type::Type;
use mlir_sys::{
    mlirValueDump, mlirValueEqual, mlirValueGetType, mlirValueIsABlockArgument,
    mlirValueIsAOpResult, MlirValue,
};
use std::marker::PhantomData;

/// A value.
// Values are always non-owning references to their parents, such as operations
// and block arguments. See the `Value` class in the MLIR C++ API.
#[derive(Clone, Copy, Debug)]
pub struct Value<'a> {
    raw: MlirValue,
    _parent: PhantomData<&'a ()>,
}

impl<'a> Value<'a> {
    /// Gets a type.
    pub fn r#type(&self) -> Type {
        unsafe { Type::from_raw(mlirValueGetType(self.raw)) }
    }

    /// Returns `true` if a value is a block argument.
    pub fn is_block_argument(&self) -> bool {
        unsafe { mlirValueIsABlockArgument(self.raw) }
    }

    /// Returns `true` if a value is an operation result.
    pub fn is_operation_result(&self) -> bool {
        unsafe { mlirValueIsAOpResult(self.raw) }
    }

    /// Dumps a value.
    pub fn dump(&self) {
        unsafe { mlirValueDump(self.raw) }
    }

    pub(crate) unsafe fn from_raw(value: MlirValue) -> Self {
        Self {
            raw: value,
            _parent: Default::default(),
        }
    }

    pub(crate) unsafe fn to_raw(self) -> MlirValue {
        self.raw
    }
}

impl<'a> PartialEq for Value<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirValueEqual(self.raw, other.raw) }
    }
}

impl<'a> Eq for Value<'a> {}

#[cfg(test)]
mod tests {
    use crate::{
        attribute::Attribute, block::Block, context::Context, identifier::Identifier,
        location::Location, operation::Operation, operation_state::OperationState, r#type::Type,
    };

    #[test]
    fn r#type() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::parse(&context, "index");

        let operation = Operation::new(
            OperationState::new("arith.constant", location)
                .add_results(&[index_type])
                .add_attributes(&[(
                    Identifier::new(&context, "value"),
                    Attribute::parse(&context, "0 : index"),
                )]),
        );

        assert_eq!(operation.result(0).unwrap().r#type(), index_type);
    }

    #[test]
    fn is_operation_result() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let r#type = Type::parse(&context, "index");

        let operation = Operation::new(
            OperationState::new("arith.constant", location)
                .add_results(&[r#type])
                .add_attributes(&[(
                    Identifier::new(&context, "value"),
                    Attribute::parse(&context, "0 : index"),
                )]),
        );

        assert!(operation.result(0).unwrap().is_operation_result());
    }

    #[test]
    fn is_block_argument() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index");
        let block = Block::new(&[(r#type, Location::unknown(&context))]);

        assert!(block.argument(0).unwrap().is_block_argument());
    }

    #[test]
    fn dump() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::parse(&context, "index");

        let value = Operation::new(
            OperationState::new("arith.constant", location)
                .add_results(&[index_type])
                .add_attributes(&[(
                    Identifier::new(&context, "value"),
                    Attribute::parse(&context, "0 : index"),
                )]),
        );

        value.result(0).unwrap().dump();
    }

    #[test]
    fn equal() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::parse(&context, "index");

        let operation = Operation::new(
            OperationState::new("arith.constant", location)
                .add_results(&[index_type])
                .add_attributes(&[(
                    Identifier::new(&context, "value"),
                    Attribute::parse(&context, "0 : index"),
                )]),
        );

        assert_eq!(operation.result(0), operation.result(0));
    }

    #[test]
    fn not_equal() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::parse(&context, "index");

        let operation = || {
            Operation::new(
                OperationState::new("arith.constant", location)
                    .add_results(&[index_type])
                    .add_attributes(&[(
                        Identifier::new(&context, "value"),
                        Attribute::parse(&context, "0 : index"),
                    )]),
            )
        };

        assert_ne!(operation().result(0), operation().result(0));
    }
}
