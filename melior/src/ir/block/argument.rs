use super::Value;
use crate::{
    ir::{BlockRef, Type, TypeLike, ValueLike},
    Error,
};
use mlir_sys::{
    mlirBlockArgumentGetArgNumber, mlirBlockArgumentGetOwner, mlirBlockArgumentSetType, MlirValue,
};
use std::fmt::{self, Display, Formatter};

/// A block argument.
#[derive(Clone, Copy, Debug)]
pub struct BlockArgument<'a> {
    value: Value<'a>,
}

impl<'a> BlockArgument<'a> {
    pub fn argument_number(&self) -> usize {
        unsafe { mlirBlockArgumentGetArgNumber(self.value.to_raw()) as usize }
    }

    pub fn owner(&self) -> BlockRef {
        unsafe { BlockRef::from_raw(mlirBlockArgumentGetOwner(self.value.to_raw())) }
    }

    pub fn set_type(&self, r#type: Type) {
        unsafe { mlirBlockArgumentSetType(self.value.to_raw(), r#type.to_raw()) }
    }

    pub unsafe fn from_raw(value: MlirValue) -> Self {
        Self {
            value: Value::from_raw(value),
        }
    }
}

impl<'a> ValueLike for BlockArgument<'a> {
    fn to_raw(&self) -> MlirValue {
        self.value.to_raw()
    }
}

impl<'a> Display for BlockArgument<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Value::from(*self).fmt(formatter)
    }
}

impl<'a> TryFrom<Value<'a>> for BlockArgument<'a> {
    type Error = Error;

    fn try_from(value: Value<'a>) -> Result<Self, Self::Error> {
        if value.is_block_argument() {
            Ok(Self { value })
        } else {
            Err(Error::BlockArgumentExpected(value.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        ir::{Block, Location},
    };

    #[test]
    fn argument_number() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);

        assert_eq!(block.argument(0).unwrap().argument_number(), 0);
    }

    #[test]
    fn owner() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);

        assert_eq!(&*block.argument(0).unwrap().owner(), &block);
    }

    #[test]
    fn set_type() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let other_type = Type::parse(&context, "f64").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);
        let argument = block.argument(0).unwrap();

        argument.set_type(other_type);

        assert_eq!(argument.r#type(), other_type);
    }
}
