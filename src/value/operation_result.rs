use super::Value;
use crate::operation::OperationRef;
use mlir_sys::{mlirOpResultGetOwner, mlirOpResultGetResultNumber};
use std::ops::Deref;

/// An operation result.
#[derive(Clone, Copy, Debug)]
pub struct OperationResult<'a> {
    value: Value<'a>,
}

impl<'a> OperationResult<'a> {
    pub fn result_number(&self) -> usize {
        unsafe { mlirOpResultGetResultNumber(self.value.to_raw()) as usize }
    }

    pub fn owner(&self) -> OperationRef {
        unsafe { OperationRef::from_raw(mlirOpResultGetOwner(self.value.to_raw())) }
    }

    pub(crate) unsafe fn from_value(value: Value<'a>) -> Self {
        Self { value }
    }
}

impl<'a> Deref for OperationResult<'a> {
    type Target = Value<'a>;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

#[cfg(test)]
mod tests {
    use crate::{block::Block, context::Context, location::Location, operation, r#type::Type};

    #[test]
    fn result_number() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let operation = operation::Builder::new("foo", Location::unknown(&context))
            .add_results(&[r#type])
            .build();

        assert_eq!(operation.result(0).unwrap().result_number(), 0);
    }

    #[test]
    fn owner() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);

        assert_eq!(block.argument(0).unwrap().owner(), *block);
    }
}
