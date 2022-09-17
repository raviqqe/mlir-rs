mod block_argument;
mod operation_result;

pub use self::{block_argument::BlockArgument, operation_result::OperationResult};
use crate::{r#type::Type, string_ref::StringRef};
use mlir_sys::{
    mlirValueDump, mlirValueEqual, mlirValueGetType, mlirValueIsABlockArgument,
    mlirValueIsAOpResult, mlirValuePrint, MlirStringRef, MlirValue,
};
use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
};

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

impl<'a> Display for Value<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe extern "C" fn callback(string: MlirStringRef, data: *mut c_void) {
            let data = &mut *(data as *mut (&mut Formatter, fmt::Result));
            let result = write!(data.0, "{}", StringRef::from_raw(string).as_str());

            if data.1.is_ok() {
                data.1 = result;
            }
        }

        unsafe {
            mlirValuePrint(self.raw, Some(callback), &mut data as *mut _ as *mut c_void);
        }

        data.1
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        attribute::Attribute, block::Block, context::Context, dialect_registry::DialectRegistry,
        identifier::Identifier, location::Location, operation, r#type::Type,
        utility::register_all_dialects,
    };

    #[test]
    fn r#type() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::parse(&context, "index").unwrap();

        let operation = operation::Builder::new("arith.constant", location)
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
        let r#type = Type::parse(&context, "index").unwrap();

        let operation = operation::Builder::new("arith.constant", location)
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
        let r#type = Type::parse(&context, "index").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);

        assert!(block.argument(0).unwrap().is_block_argument());
    }

    #[test]
    fn dump() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::parse(&context, "index").unwrap();

        let value = operation::Builder::new("arith.constant", location)
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
        let index_type = Type::parse(&context, "index").unwrap();

        let operation = operation::Builder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();
        let result = *operation.result(0).unwrap();

        assert_eq!(result, result);
    }

    #[test]
    fn not_equal() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::parse(&context, "index").unwrap();

        let operation = || {
            operation::Builder::new("arith.constant", location)
                .add_results(&[index_type])
                .add_attributes(&[(
                    Identifier::new(&context, "value"),
                    Attribute::parse(&context, "0 : index").unwrap(),
                )])
                .build()
        };

        assert_ne!(
            *operation().result(0).unwrap(),
            *operation().result(0).unwrap()
        );
    }

    #[test]
    fn display() {
        let context = Context::new();
        context.load_all_available_dialects();
        let location = Location::unknown(&context);
        let index_type = Type::parse(&context, "index").unwrap();

        let operation = operation::Builder::new("arith.constant", location)
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
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let location = Location::unknown(&context);
        let index_type = Type::parse(&context, "index").unwrap();

        let operation = operation::Builder::new("arith.constant", location)
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
}
