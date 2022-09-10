use mlir_sys::{mlirStringRefCreateFromCString, MlirStringRef};
use std::{
    ffi::{CStr, CString},
    slice, str,
};

pub struct StringRef {
    string: MlirStringRef,
}

// https://mlir.llvm.org/docs/CAPI/#stringref
//
// TODO Handle non-null terminated strings.
impl StringRef {
    pub fn as_str(&self) -> &CStr {
        unsafe {
            CStr::from_bytes_with_nul(slice::from_raw_parts(
                self.string.data as *mut u8,
                self.string.length as usize,
            ))
            .unwrap()
        }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirStringRef {
        self.string
    }

    pub(crate) unsafe fn from_raw(string: MlirStringRef) -> Self {
        Self { string }
    }
}

impl From<&str> for StringRef {
    fn from(string: &str) -> Self {
        unsafe {
            Self::from_raw(mlirStringRefCreateFromCString(
                CString::new(string).unwrap().into_raw(),
            ))
        }
    }
}
