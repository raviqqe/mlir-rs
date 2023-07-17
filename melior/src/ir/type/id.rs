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
    /// Creates a type ID from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub const unsafe fn from_raw(raw: MlirTypeID) -> Self {
        Self { raw }
    }

    /// Converts a type ID into a raw object.
    pub const fn to_raw(self) -> MlirTypeID {
        self.raw
    }

    /// Creates a type ID from an 8-byte aligned reference.
    ///
    /// # Panics
    ///
    /// This function will panic if the given reference is not 8-byte aligned.
    pub fn create<T>(t: &T) -> Self {
        unsafe {
            let ptr = t as *const _ as *const std::ffi::c_void;
            assert_eq!(
                ptr.align_offset(8),
                0,
                "type ID pointer must be 8-byte aligned"
            );
            Self::from_raw(mlir_sys::mlirTypeIDCreate(ptr))
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
