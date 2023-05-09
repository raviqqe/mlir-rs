use crate::ContextRef;

use super::r#type;
use super::Type;
use mlir_sys::{mlirAttributeDump, mlirAttributeGetType, mlirAttributeGetTypeID, MlirAttribute};
use mlir_sys::{
    mlirAttributeGetContext, mlirAttributeIsAAffineMap, mlirAttributeIsAArray,
    mlirAttributeIsABool, mlirAttributeIsADenseElements, mlirAttributeIsADenseFPElements,
    mlirAttributeIsADenseIntElements, mlirAttributeIsADictionary, mlirAttributeIsAElements,
    mlirAttributeIsAFloat, mlirAttributeIsAInteger, mlirAttributeIsAIntegerSet,
    mlirAttributeIsAOpaque, mlirAttributeIsASparseElements, mlirAttributeIsAString,
    mlirAttributeIsASymbolRef, mlirAttributeIsAType, mlirAttributeIsAUnit,
};

/// Trait for attribute-like types.
pub trait AttributeLike<'c> {
    /// Converts a attribute into a raw attribute.
    unsafe fn to_raw(&self) -> MlirAttribute;

    /// Gets a context.
    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAttributeGetContext(self.to_raw())) }
    }

    /// Gets a type.
    fn r#type(&self) -> Type {
        unsafe { Type::from_raw(mlirAttributeGetType(self.to_raw())) }
    }

    /// Gets a type ID.
    fn type_id(&self) -> Option<r#type::Id> {
        if self.is_null() {
            None
        } else {
            unsafe { Some(r#type::Id::from_raw(mlirAttributeGetTypeID(self.to_raw()))) }
        }
    }

    /// Returns `true` if an attribute is null.
    fn is_null(&self) -> bool {
        self.to_raw().ptr.is_null()
    }

    /// Returns `true` if an attribute is a affine map.
    fn is_affine_map(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsAAffineMap(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a array.
    fn is_array(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsAArray(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a bool.
    fn is_bool(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsABool(self.to_raw()) }
    }

    /// Returns `true` if an attribute is dense elements.
    fn is_dense_elements(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsADenseElements(self.to_raw()) }
    }

    /// Returns `true` if an attribute is dense integer elements.
    fn is_dense_integer_elements(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsADenseIntElements(self.to_raw()) }
    }

    /// Returns `true` if an attribute is dense float elements.
    fn is_dense_float_elements(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsADenseFPElements(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a dictionary.
    fn is_dictionary(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsADictionary(self.to_raw()) }
    }

    /// Returns `true` if an attribute is elements.
    fn is_elements(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsAElements(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a float.
    fn is_float(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsAFloat(self.to_raw()) }
    }

    /// Returns `true` if an attribute is an integer.
    fn is_integer(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsAInteger(self.to_raw()) }
    }

    /// Returns `true` if an attribute is an integer set.
    fn is_integer_set(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsAIntegerSet(self.to_raw()) }
    }

    /// Returns `true` if an attribute is opaque.
    fn is_opaque(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsAOpaque(self.to_raw()) }
    }

    /// Returns `true` if an attribute is sparse elements.
    fn is_sparse_elements(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsASparseElements(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a string.
    fn is_string(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsAString(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a symbol.
    fn is_symbol(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsASymbolRef(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a type.
    fn is_type(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsAType(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a unit.
    fn is_unit(&self) -> bool {
        !self.is_null() && unsafe { mlirAttributeIsAUnit(self.to_raw()) }
    }

    /// Dumps a attribute.
    fn dump(&self) {
        unsafe { mlirAttributeDump(self.to_raw()) }
    }
}
