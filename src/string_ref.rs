use mlir_sys::{mlirStringRefCreateFromCString, MlirStringRef};
use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    slice, str,
};

// https://mlir.llvm.org/docs/CAPI/#stringref
pub struct StringRef<'a> {
    string: MlirStringRef,
    _parent: PhantomData<&'a ()>,
}

// TODO Handle non-null terminated strings.
impl<'a> StringRef<'a> {
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
        Self {
            string,
            _parent: Default::default(),
        }
    }
}

impl From<&str> for StringRef<'_> {
    fn from(string: &str) -> Self {
        let string = CString::new(string).unwrap();

        unsafe { Self::from_raw(mlirStringRefCreateFromCString(string.as_ptr())) }
    }
}
