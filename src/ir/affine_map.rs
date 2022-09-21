use crate::{
    context::{Context, ContextRef},
    utility::print_callback,
};
use mlir_sys::{
    mlirAffineMapDump, mlirAffineMapEqual, mlirAffineMapGetContext, mlirAffineMapPrint,
    MlirAffineMap,
};
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

/// An affine map.
#[derive(Clone, Copy)]
pub struct AffineMap<'c> {
    raw: MlirAffineMap,
    _context: PhantomData<&'c Context>,
}

impl<'c> AffineMap<'c> {
    /// Gets a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAffineMapGetContext(self.raw)) }
    }

    /// Dumps an attribute.
    pub fn dump(&self) {
        unsafe { mlirAffineMapDump(self.raw) }
    }

    pub(crate) unsafe fn from_raw(raw: MlirAffineMap) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
}

impl<'c> PartialEq for AffineMap<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirAffineMapEqual(self.raw, other.raw) }
    }
}

impl<'c> Eq for AffineMap<'c> {}

impl<'c> Display for AffineMap<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirAffineMapPrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl<'c> Debug for AffineMap<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}
