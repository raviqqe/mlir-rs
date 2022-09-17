use super::Value;
use crate::{
    ir::{BlockRef, Type, ValueLike},
    utility::print_callback,
    Error,
};
use mlir_sys::{
    mlirBlockArgumentGetArgNumber, mlirBlockArgumentGetOwner, mlirBlockArgumentSetType,
    mlirValuePrint, MlirValue,
};
use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
};

/// A block argument.
#[derive(Clone, Copy, Debug)]
pub struct Argument<'a> {
    value: Value<'a>,
}

impl<'a> Argument<'a> {
    pub fn argument_number(&self) -> usize {
        unsafe { mlirBlockArgumentGetArgNumber(self.value.to_raw()) as usize }
    }

    pub fn owner(&self) -> BlockRef {
        unsafe { BlockRef::from_raw(mlirBlockArgumentGetOwner(self.value.to_raw())) }
    }

    pub fn set_type(&self, r#type: Type) {
        unsafe { mlirBlockArgumentSetType(self.value.to_raw(), r#type.to_raw()) }
    }
}

impl<'a> ValueLike for Argument<'a> {
    unsafe fn from_raw(value: MlirValue) -> Self {
        Self {
            value: Value::from_raw(value),
        }
    }

    unsafe fn to_raw(&self) -> MlirValue {
        self.value.to_raw()
    }
}

impl<'a> Display for Argument<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirValuePrint(
                self.value.to_raw(),
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl<'a> TryFrom<Value<'a>> for Argument<'a> {
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

        assert_eq!(block.argument(0).unwrap().owner(), *block);
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
