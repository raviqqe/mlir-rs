//! Type IDs and allocators

mod allocator;

pub use allocator::Allocator;
use mlir_sys::{mlirTypeIDEqual, mlirTypeIDHashValue, MlirTypeID};
use std::hash::{Hash, Hasher};

/// A type ID.
#[derive(Clone, Copy, Debug)]
pub struct TypeId {
    raw: MlirTypeID,
}

impl TypeId {
    pub(crate) unsafe fn from_raw(raw: MlirTypeID) -> Self {
        Self { raw }
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
