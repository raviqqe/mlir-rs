use super::TypeLike;
use crate::{ir::Type, utility::into_raw_array, Context, Error};
use mlir_sys::{
    mlirTupleTypeGet, mlirTupleTypeGetInput, mlirTupleTypeGetNumInputs, mlirTupleTypeGetNumResults,
    mlirTupleTypeGetNumTypes, mlirTupleTypeGetResult, MlirType,
};
use std::fmt::{self, Display, Formatter};

/// A tuple type.
#[derive(Clone, Copy, Debug)]
pub struct Tuple<'c> {
    r#type: Type<'c>,
}

impl<'c> Tuple<'c> {
    /// Creates a tuple type.
    pub fn new(context: &'c Context, types: &[Type<'c>]) -> Self {
        Self {
            r#type: unsafe {
                Type::from_raw(mlirTupleTypeGet(
                    context.to_raw(),
                    types.len() as isize,
                    into_raw_array(types.iter().map(|r#type| r#type.to_raw()).collect()),
                ))
            },
        }
    }

    /// Gets a field at a position.
    pub fn r#type(&self, position: usize) -> Result<Type, Error> {
        if position < self.type_count() {
            unsafe {
                Ok(Type::from_raw(mlirTupleTypeGetInput(
                    self.r#type.to_raw(),
                    position as isize,
                )))
            }
        } else {
            Err(Error::TupleFieldPosition(self.to_string(), position))
        }
    }

    /// Gets a number of fields.
    pub fn type_count(&self) -> usize {
        unsafe { mlirTupleTypeGetNumTypes(self.r#type.to_raw()) as usize }
    }
}

impl<'c> TypeLike<'c> for Tuple<'c> {
    fn to_raw(&self) -> MlirType {
        self.r#type.to_raw()
    }
}

impl<'c> Display for Tuple<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Type::from(*self).fmt(formatter)
    }
}

impl<'c> TryFrom<Type<'c>> for Tuple<'c> {
    type Error = Error;

    fn try_from(r#type: Type<'c>) -> Result<Self, Self::Error> {
        if r#type.is_tuple() {
            Ok(Self { r#type })
        } else {
            Err(Error::TupleExpected(r#type.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn new() {
        let context = Context::new();
        let integer = Type::integer(&context, 42);

        assert_eq!(
            Type::from(Tuple::new(&context, &[integer, integer], &[integer])),
            Type::parse(&context, "(i42, i42) -> i42").unwrap()
        );
    }
}
