use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{
    mlirAttributeDump, mlirAttributeEqual, mlirAttributeGetContext, mlirAttributeGetNull,
    mlirAttributeParseGet, mlirAttributePrint, MlirAttribute, MlirStringRef,
};
use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
};

/// An attribute.
// Attributes are always values but their internal storage is owned by contexts.
#[derive(Clone, Copy, Debug)]
pub struct Attribute<'c> {
    raw: MlirAttribute,
    _context: PhantomData<&'c Context>,
}

impl<'c> Attribute<'c> {
    /// Parses an attribute.
    pub fn parse(context: &'c Context, source: &str) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirAttributeParseGet(
                context.to_raw(),
                StringRef::from(source).to_raw(),
            ))
        }
    }

    /// Creates a null attribute.
    pub fn null() -> Self {
        unsafe { Self::from_raw(mlirAttributeGetNull()) }
    }

    /// Gets a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAttributeGetContext(self.raw)) }
    }

    /// Dumps an attribute.
    pub fn dump(&self) {
        unsafe { mlirAttributeDump(self.raw) }
    }

    unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    unsafe fn from_option_raw(raw: MlirAttribute) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }

    pub(crate) unsafe fn to_raw(self) -> MlirAttribute {
        self.raw
    }
}

impl<'c> PartialEq for Attribute<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirAttributeEqual(self.raw, other.raw) }
    }
}

impl<'c> Eq for Attribute<'c> {}

impl<'c> Display for Attribute<'c> {
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
            mlirAttributePrint(self.raw, Some(callback), &mut data as *mut _ as *mut c_void);
        }

        data.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        for attribute in ["unit", "i32", r#""foo""#] {
            assert!(Attribute::parse(&Context::new(), attribute).is_some());
        }
    }

    #[test]
    fn parse_none() {
        assert!(Attribute::parse(&Context::new(), "z").is_none());
    }

    #[test]
    fn null() {
        assert_eq!(Attribute::null().to_string(), "null");
    }

    #[test]
    fn context() {
        Attribute::parse(&Context::new(), "unit").unwrap().context();
    }

    #[test]
    fn equal() {
        let context = Context::new();
        let attribute = Attribute::parse(&context, "unit").unwrap();

        assert_eq!(attribute, attribute);
    }

    #[test]
    fn not_equal() {
        let context = Context::new();

        assert_ne!(
            Attribute::parse(&context, "unit").unwrap(),
            Attribute::parse(&context, "42").unwrap()
        );
    }

    #[test]
    fn display() {
        assert_eq!(
            Attribute::parse(&Context::new(), "unit")
                .unwrap()
                .to_string(),
            "unit"
        );
    }
}
