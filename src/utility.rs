use crate::string_ref::StringRef;
use mlir_sys::MlirStringRef;
use std::mem::forget;
use std::ptr::null_mut;

pub(crate) unsafe fn as_string_ref(string: &str) -> MlirStringRef {
    StringRef::from(string).to_raw()
}

pub(crate) unsafe fn into_raw_array<T>(mut xs: Vec<T>) -> *mut T {
    if xs.is_empty() {
        null_mut()
    } else {
        let ptr = xs.as_mut_ptr();

        forget(xs);

        ptr
    }
}
