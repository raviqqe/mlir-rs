use super::{r#type, Type};
use crate::ContextRef;
use mlir_sys::{
    mlirAttributeDump, mlirAttributeGetContext, mlirAttributeGetType, mlirAttributeGetTypeID,
    mlirAttributeIsAAffineMap, mlirAttributeIsAArray, mlirAttributeIsABool,
    mlirAttributeIsADenseElements, mlirAttributeIsADenseFPElements,
    mlirAttributeIsADenseIntElements, mlirAttributeIsADictionary, mlirAttributeIsAElements,
    mlirAttributeIsAFloat, mlirAttributeIsAInteger, mlirAttributeIsAIntegerSet,
    mlirAttributeIsAOpaque, mlirAttributeIsASparseElements, mlirAttributeIsAString,
    mlirAttributeIsASymbolRef, mlirAttributeIsAType, mlirAttributeIsAUnit, MlirAttribute,
};

/// Trait for attribute-like types.
pub trait AttributeLike<'c> {
    /// Converts a attribute into a raw attribute.
    fn to_raw(&self) -> MlirAttribute;

    /// Gets a context.
    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAttributeGetContext(self.to_raw())) }
    }

    /// Gets a type.
    fn r#type(&self) -> Type {
        unsafe { Type::from_raw(mlirAttributeGetType(self.to_raw())) }
    }

    /// Gets a type ID.
    fn type_id(&self) -> r#type::Id {
        unsafe { r#type::Id::from_raw(mlirAttributeGetTypeID(self.to_raw())) }
    }

    /// Returns `true` if an attribute is a affine map.
    fn is_affine_map(&self) -> bool {
        unsafe { mlirAttributeIsAAffineMap(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a array.
    fn is_array(&self) -> bool {
        unsafe { mlirAttributeIsAArray(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a bool.
    fn is_bool(&self) -> bool {
        unsafe { mlirAttributeIsABool(self.to_raw()) }
    }

    /// Returns `true` if an attribute is dense elements.
    fn is_dense_elements(&self) -> bool {
        unsafe { mlirAttributeIsADenseElements(self.to_raw()) }
    }

    /// Returns `true` if an attribute is dense integer elements.
    fn is_dense_integer_elements(&self) -> bool {
        unsafe { mlirAttributeIsADenseIntElements(self.to_raw()) }
    }

    /// Returns `true` if an attribute is dense float elements.
    fn is_dense_float_elements(&self) -> bool {
        unsafe { mlirAttributeIsADenseFPElements(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a dictionary.
    fn is_dictionary(&self) -> bool {
        unsafe { mlirAttributeIsADictionary(self.to_raw()) }
    }

    /// Returns `true` if an attribute is elements.
    fn is_elements(&self) -> bool {
        unsafe { mlirAttributeIsAElements(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a float.
    fn is_float(&self) -> bool {
        unsafe { mlirAttributeIsAFloat(self.to_raw()) }
    }

    /// Returns `true` if an attribute is an integer.
    fn is_integer(&self) -> bool {
        unsafe { mlirAttributeIsAInteger(self.to_raw()) }
    }

    /// Returns `true` if an attribute is an integer set.
    fn is_integer_set(&self) -> bool {
        unsafe { mlirAttributeIsAIntegerSet(self.to_raw()) }
    }

    /// Returns `true` if an attribute is opaque.
    fn is_opaque(&self) -> bool {
        unsafe { mlirAttributeIsAOpaque(self.to_raw()) }
    }

    /// Returns `true` if an attribute is sparse elements.
    fn is_sparse_elements(&self) -> bool {
        unsafe { mlirAttributeIsASparseElements(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a string.
    fn is_string(&self) -> bool {
        unsafe { mlirAttributeIsAString(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a symbol.
    fn is_symbol(&self) -> bool {
        unsafe { mlirAttributeIsASymbolRef(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a type.
    fn is_type(&self) -> bool {
        unsafe { mlirAttributeIsAType(self.to_raw()) }
    }

    /// Returns `true` if an attribute is a unit.
    fn is_unit(&self) -> bool {
        unsafe { mlirAttributeIsAUnit(self.to_raw()) }
    }

    /// Dumps a attribute.
    fn dump(&self) {
        unsafe { mlirAttributeDump(self.to_raw()) }
    }
}
