use std::mem::forget;

pub(crate) unsafe fn as_string_ref(string: &str) -> mlir_sys::MlirStringRef {
    let string = string.as_bytes();

    mlir_sys::MlirStringRef {
        data: string.as_ptr() as *const i8,
        length: string.len() as u64,
    }
}

pub(crate) unsafe fn into_raw_array<T>(xs: Vec<T>) -> *mut T {
    let ptr = xs.as_mut_ptr();

    forget(xs);

    ptr
}
