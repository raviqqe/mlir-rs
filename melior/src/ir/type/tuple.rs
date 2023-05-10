use super::TypeLike;
use crate::{ir::Type, utility::into_raw_array, Context, Error};
use mlir_sys::{mlirTupleTypeGet, mlirTupleTypeGetNumTypes, mlirTupleTypeGetType, MlirType};
use std::fmt::{self, Display, Formatter};

/// A tuple type.
#[derive(Clone, Copy, Debug)]
pub struct TupleType<'c> {
    r#type: Type<'c>,
}

impl<'c> TupleType<'c> {
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
                Ok(Type::from_raw(mlirTupleTypeGetType(
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

impl<'c> TypeLike<'c> for TupleType<'c> {
    fn to_raw(&self) -> MlirType {
        self.r#type.to_raw()
    }
}

impl<'c> Display for TupleType<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Type::from(*self).fmt(formatter)
    }
}

impl<'c> TryFrom<Type<'c>> for TupleType<'c> {
    type Error = Error;

    fn try_from(r#type: Type<'c>) -> Result<Self, Self::Error> {
        if r#type.is_tuple() {
            Ok(Self { r#type })
        } else {
            Err(Error::TypeExpected("tuple", r#type.to_string()))
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

        assert_eq!(
            Type::from(TupleType::new(&context, &[])),
            Type::parse(&context, "tuple<>").unwrap()
        );
    }

    #[test]
    fn new_with_field() {
        let context = Context::new();

        assert_eq!(
            Type::from(TupleType::new(&context, &[Type::index(&context)])),
            Type::parse(&context, "tuple<index>").unwrap()
        );
    }

    #[test]
    fn new_with_two_fields() {
        let context = Context::new();
        let r#type = Type::index(&context);

        assert_eq!(
            Type::from(TupleType::new(&context, &[r#type, r#type])),
            Type::parse(&context, "tuple<index,index>").unwrap()
        );
    }

    #[test]
    fn r#type() {
        let context = Context::new();
        let r#type = Type::index(&context);

        assert_eq!(TupleType::new(&context, &[r#type]).r#type(0), Ok(r#type));
    }

    #[test]
    fn type_error() {
        let context = Context::new();
        let tuple = TupleType::new(&context, &[]);

        assert_eq!(
            tuple.r#type(42),
            Err(Error::TupleFieldPosition(tuple.to_string(), 42))
        );
    }

    #[test]
    fn type_count() {
        assert_eq!(TupleType::new(&Context::new(), &[]).type_count(), 0);
    }
}
