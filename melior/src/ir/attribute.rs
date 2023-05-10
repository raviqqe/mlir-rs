//! Attributes.

#[macro_use]
mod r#macro;
mod attribute_like;
mod dense_i64_array;
mod float;
mod integer;
mod string;
mod r#type;

pub use self::{
    attribute_like::AttributeLike, dense_i64_array::DenseI64ArrayAttribute, float::FloatAttribute,
    integer::IntegerAttribute, r#type::TypeAttribute, string::StringAttribute,
};
use crate::{context::Context, string_ref::StringRef, utility::print_callback};
use mlir_sys::{
    mlirAttributeEqual, mlirAttributeGetNull, mlirAttributeParseGet, mlirAttributePrint,
    mlirUnitAttrGet, MlirAttribute,
};
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

/// An attribute.
// Attributes are always values but their internal storage is owned by contexts.
#[derive(Clone, Copy)]
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

    /// Creates a unit attribute.
    pub fn unit(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirUnitAttrGet(context.to_raw())) }
    }

    pub(crate) unsafe fn null() -> Self {
        unsafe { Self::from_raw(mlirAttributeGetNull()) }
    }

    pub(crate) unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    pub(crate) unsafe fn from_option_raw(raw: MlirAttribute) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}

impl<'c> AttributeLike<'c> for Attribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
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

        unsafe {
            mlirAttributePrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl<'c> Debug for Attribute<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}

impl<'c> From<DenseI64ArrayAttribute<'c>> for Attribute<'c> {
    fn from(attribute: DenseI64ArrayAttribute<'c>) -> Self {
        unsafe { Self::from_raw(attribute.to_raw()) }
    }
}

impl<'c> From<FloatAttribute<'c>> for Attribute<'c> {
    fn from(attribute: FloatAttribute<'c>) -> Self {
        unsafe { Self::from_raw(attribute.to_raw()) }
    }
}

impl<'c> From<IntegerAttribute<'c>> for Attribute<'c> {
    fn from(attribute: IntegerAttribute<'c>) -> Self {
        unsafe { Self::from_raw(attribute.to_raw()) }
    }
}

impl<'c> From<StringAttribute<'c>> for Attribute<'c> {
    fn from(attribute: StringAttribute<'c>) -> Self {
        unsafe { Self::from_raw(attribute.to_raw()) }
    }
}

impl<'c> From<TypeAttribute<'c>> for Attribute<'c> {
    fn from(attribute: TypeAttribute<'c>) -> Self {
        unsafe { Self::from_raw(attribute.to_raw()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Type, TypeLike};

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
    fn context() {
        Attribute::parse(&Context::new(), "unit").unwrap().context();
    }

    #[test]
    fn r#type() {
        let context = Context::new();

        assert_eq!(
            Attribute::parse(&context, "unit").unwrap().r#type(),
            Type::none(&context)
        );
    }

    // TODO Fix this.
    #[ignore]
    #[test]
    fn type_id() {
        let context = Context::new();

        assert_eq!(
            Attribute::parse(&context, "42 : index").unwrap().type_id(),
            Type::index(&context).id()
        );
    }

    #[test]
    fn is_array() {
        assert!(Attribute::parse(&Context::new(), "[]").unwrap().is_array());
    }

    #[test]
    fn is_bool() {
        assert!(Attribute::parse(&Context::new(), "false")
            .unwrap()
            .is_bool());
    }

    #[test]
    fn is_dense_elements() {
        assert!(
            Attribute::parse(&Context::new(), "dense<10> : tensor<2xi8>")
                .unwrap()
                .is_dense_elements()
        );
    }

    #[test]
    fn is_dense_int_elements() {
        assert!(
            Attribute::parse(&Context::new(), "dense<42> : tensor<42xi8>")
                .unwrap()
                .is_dense_int_elements()
        );
    }

    #[test]
    fn is_dense_fp_elements() {
        assert!(
            Attribute::parse(&Context::new(), "dense<42.0> : tensor<42xf32>")
                .unwrap()
                .is_dense_fp_elements()
        );
    }

    #[test]
    fn is_elements() {
        assert!(Attribute::parse(
            &Context::new(),
            "sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>"
        )
        .unwrap()
        .is_elements());
    }

    #[test]
    fn is_integer() {
        assert!(Attribute::parse(&Context::new(), "42")
            .unwrap()
            .is_integer());
    }

    #[test]
    fn is_integer_set() {
        assert!(
            Attribute::parse(&Context::new(), "affine_set<(d0) : (d0 - 2 >= 0)>")
                .unwrap()
                .is_integer_set()
        );
    }

    // TODO Fix this.
    #[ignore]
    #[test]
    fn is_opaque() {
        assert!(Attribute::parse(&Context::new(), "#foo<\"bar\">")
            .unwrap()
            .is_opaque());
    }

    #[test]
    fn is_sparse_elements() {
        assert!(Attribute::parse(
            &Context::new(),
            "sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>"
        )
        .unwrap()
        .is_sparse_elements());
    }

    #[test]
    fn is_string() {
        assert!(Attribute::parse(&Context::new(), "\"foo\"")
            .unwrap()
            .is_string());
    }

    #[test]
    fn is_type() {
        assert!(Attribute::parse(&Context::new(), "index")
            .unwrap()
            .is_type());
    }

    #[test]
    fn is_unit() {
        assert!(Attribute::parse(&Context::new(), "unit").unwrap().is_unit());
    }

    #[test]
    fn is_symbol() {
        assert!(Attribute::parse(&Context::new(), "@foo")
            .unwrap()
            .is_symbol_ref());
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
