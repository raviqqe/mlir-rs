use mlir_sys::{mlirStringRefCreateFromCString, MlirStringRef};
use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    marker::PhantomData,
    slice, str,
    sync::RwLock,
};

static STRING_CACHE: Lazy<RwLock<HashMap<String, CString>>> = Lazy::new(|| Default::default());

// https://mlir.llvm.org/docs/CAPI/#stringref
//
// TODO The documentation says string refs do not have to be null-terminated.
// But it looks like some APIs do not handle strings not null-terminated?
pub struct StringRef<'a> {
    string: MlirStringRef,
    _parent: PhantomData<&'a ()>,
}

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
        if !STRING_CACHE.read().unwrap().contains_key(string) {
            STRING_CACHE
                .write()
                .unwrap()
                .insert(string.to_owned(), CString::new(string).unwrap());
        }

        let lock = STRING_CACHE.read().unwrap();
        let string = lock.get(string).unwrap();

        unsafe { Self::from_raw(mlirStringRefCreateFromCString(string.as_ptr())) }
    }
}
