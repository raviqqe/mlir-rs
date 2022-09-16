mod allocator;

pub use allocator::Allocator;
use mlir_sys::{mlirTypeIDCreate, mlirTypeIDEqual, MlirTypeID};
use std::ffi::c_void;

/// A type ID.
#[derive(Clone, Copy, Debug)]
pub struct Id {
    raw: MlirTypeID,
}

impl Id {
    pub unsafe fn new(ptr: *const ()) -> Self {
        Self::from_raw(mlirTypeIDCreate(ptr as *const c_void))
    }

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
