pub(crate) unsafe fn as_string_ref(string: &str) -> mlir_sys::MlirStringRef {
    let string = string.as_bytes();

    mlir_sys::MlirStringRef {
        data: string.as_ptr() as *const i8,
        length: string.len() as u64,
    }
}
