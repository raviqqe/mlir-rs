//! Type IDs and allocators

mod allocator;

pub use allocator::Allocator;
use mlir_sys::{
    mlirTypeIDAllocatorAllocateTypeID, mlirTypeIDAllocatorCreate, mlirTypeIDAllocatorDestroy,
    mlirTypeIDEqual, mlirTypeIDHashValue, MlirTypeID,
};
use std::hash::{Hash, Hasher};

/// A type ID.
#[derive(Clone, Copy, Debug)]
pub struct TypeId {
    raw: MlirTypeID,
}

impl TypeId {
    /// Creates a type ID from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub const unsafe fn from_raw(raw: MlirTypeID) -> Self {
        Self { raw }
    }

    pub unsafe fn to_raw(self) -> MlirTypeID {
        self.raw
    }

    pub fn create() -> Self {
        unsafe {
            let allocator = mlirTypeIDAllocatorCreate();
            let id = mlirTypeIDAllocatorAllocateTypeID(allocator);
            mlirTypeIDAllocatorDestroy(allocator);
            Self::from_raw(id)
        }
    }
}

impl PartialEq for TypeId {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirTypeIDEqual(self.raw, other.raw) }
    }
}

impl Eq for TypeId {}

impl Hash for TypeId {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        unsafe {
            mlirTypeIDHashValue(self.raw).hash(hasher);
        }
    }
}
