use super::Value;
use crate::{
    ir::{OperationRef, ValueLike},
    Error,
};
use mlir_sys::{mlirOpResultGetOwner, mlirOpResultGetResultNumber, MlirValue};
use std::fmt::{self, Display, Formatter};

/// An operation result.
#[derive(Clone, Copy, Debug)]
pub struct OperationResult<'a> {
    value: Value<'a>,
}

impl<'a> OperationResult<'a> {
    /// Gets a result number.
    pub fn result_number(&self) -> usize {
        unsafe { mlirOpResultGetResultNumber(self.value.to_raw()) as usize }
    }

    /// Gets an owner operation.
    pub fn owner(&self) -> OperationRef {
        unsafe { OperationRef::from_raw(mlirOpResultGetOwner(self.value.to_raw())) }
    }

    /// Creates an operation result from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(value: MlirValue) -> Self {
        Self {
            value: Value::from_raw(value),
        }
    }
}

impl<'a> ValueLike for OperationResult<'a> {
    fn to_raw(&self) -> MlirValue {
        self.value.to_raw()
    }
}

impl<'a> Display for OperationResult<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Value::from(*self).fmt(formatter)
    }
}

impl<'a> TryFrom<Value<'a>> for OperationResult<'a> {
    type Error = Error;

    fn try_from(value: Value<'a>) -> Result<Self, Self::Error> {
        if value.is_operation_result() {
            Ok(Self { value })
        } else {
            Err(Error::OperationResultExpected(value.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ir::{operation::OperationBuilder, Block, Location, Type},
        test::create_test_context,
    };

    #[test]
    fn result_number() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let r#type = Type::parse(&context, "index").unwrap();
        let operation = OperationBuilder::new("foo", Location::unknown(&context))
            .add_results(&[r#type])
            .build();

        assert_eq!(operation.result(0).unwrap().result_number(), 0);
    }

    #[test]
    fn owner() {
        let context = create_test_context();
        let r#type = Type::parse(&context, "index").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);

        assert_eq!(&*block.argument(0).unwrap().owner(), &block);
    }
}
