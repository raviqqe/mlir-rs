mod allocator;

pub use allocator::Allocator;
use mlir_sys::{mlirTypeIDEqual, mlirTypeIDHashValue, MlirTypeID};
use std::hash::{Hash, Hasher};

/// A type ID.
#[derive(Clone, Copy, Debug)]
pub struct Id {
    raw: MlirTypeID,
}

impl Id {
    pub(crate) unsafe fn from_raw(raw: MlirTypeID) -> Self {
        Self { raw }
    }
}

impl PartialEq for Id {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirTypeIDEqual(self.raw, other.raw) }
    }
}

impl Eq for Id {}

impl Hash for Id {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        unsafe {
            mlirTypeIDHashValue(self.raw).hash(hasher);
        }
    }
}
