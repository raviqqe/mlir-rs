use std::ffi::CString;
use std::mem::forget;
use std::ptr::null_mut;

pub(crate) unsafe fn as_string_ref(string: &str) -> mlir_sys::MlirStringRef {
    let length = string.len() as u64;

    mlir_sys::MlirStringRef {
        data: CString::new(string).unwrap().into_raw(),
        length,
    }
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
