use mlir_sys::{mlirStringRefCreateFromCString, MlirStringRef};
use std::{ffi::CString, slice, str};

pub struct StringRef {
    string: MlirStringRef,
}

impl StringRef {
    pub fn as_str(&self) -> &str {
        unsafe {
            str::from_utf8_unchecked(slice::from_raw_parts(
                self.string.data as *mut u8,
                self.string.length as usize,
            ))
        }
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
