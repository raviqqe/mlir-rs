mod value_like;

pub use self::value_like::ValueLike;
use super::{block::Argument, operation::ResultValue, Type};
use crate::utility::print_callback;
use mlir_sys::{mlirValueEqual, mlirValuePrint, MlirValue};
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

/// A value.
// Values are always non-owning references to their parents, such as operations
// and block arguments. See the `Value` class in the MLIR C++ API.
#[derive(Clone, Copy)]
pub struct Value<'a> {
    raw: MlirValue,
    _parent: PhantomData<&'a ()>,
}

impl<'a> Value<'a> {
    pub(crate) unsafe fn from_raw(value: MlirValue) -> Self {
        Self {
            raw: value,
            _parent: Default::default(),
        }
    }
}

impl<'a> ValueLike for Value<'a> {
    fn to_raw(&self) -> MlirValue {
        self.raw
    }
}

impl<'a> PartialEq for Value<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirValueEqual(self.raw, other.raw) }
    }
}

impl<'a> Eq for Value<'a> {}

impl<'a> Display for Value<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirValuePrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl<'a> Debug for Value<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        writeln!(formatter, "Value(")?;
        Display::fmt(self, formatter)?;
        write!(formatter, ")")
    }
}

from_raw_subtypes!(Value, Argument, ResultValue);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        ir::{operation::OperationBuilder, Attribute, Block, Identifier, Location},
        test::load_all_dialects,
    };

    #[test]
    fn r#type() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();

        assert_eq!(operation.result(0).unwrap().r#type(), index_type);
    }

    #[test]
    fn is_operation_result() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let r#type = Type::index(&context);

        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[r#type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();

        assert!(operation.result(0).unwrap().is_operation_result());
    }

    #[test]
    fn is_block_argument() {
        let context = Context::new();
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, Location::unknown(&context))]);

        assert!(block.argument(0).unwrap().is_block_argument());
    }

    #[test]
    fn dump() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let value = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();

        value.result(0).unwrap().dump();
    }

    #[test]
    fn equal() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();
        let result = Value::from(operation.result(0).unwrap());

        assert_eq!(result, result);
    }

    #[test]
    fn not_equal() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let operation = || {
            OperationBuilder::new("arith.constant", location)
                .add_results(&[index_type])
                .add_attributes(&[(
                    Identifier::new(&context, "value"),
                    Attribute::parse(&context, "0 : index").unwrap(),
                )])
                .build()
        };

        assert_ne!(
            Value::from(operation().result(0).unwrap()),
            operation().result(0).unwrap().into()
        );
    }

    #[test]
    fn display() {
        let context = Context::new();

        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();

        assert_eq!(
            operation.result(0).unwrap().to_string(),
            "%0 = \"arith.constant\"() {value = 0 : index} : () -> index\n"
        );
    }

    #[test]
    fn display_with_dialect_loaded() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();

        assert_eq!(
            operation.result(0).unwrap().to_string(),
            "%c0 = arith.constant 0 : index\n"
        );
    }

    #[test]
    fn debug() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();

        assert_eq!(
            format!("{:?}", Value::from(operation.result(0).unwrap())),
            "Value(\n%c0 = arith.constant 0 : index\n)"
        );
    }
}
