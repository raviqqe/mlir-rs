use crate::Context;
use mlir_sys::{mlirStringRefCreateFromCString, mlirStringRefEqual, MlirStringRef};
use std::{
    ffi::CString,
    marker::PhantomData,
    slice,
    str::{self, Utf8Error},
};

/// A string reference.
// https://mlir.llvm.org/docs/CAPI/#stringref
//
// TODO The documentation says string refs do not have to be null-terminated.
// But it looks like some functions do not handle strings not null-terminated?
#[derive(Clone, Copy, Debug)]
pub struct StringRef<'c> {
    raw: MlirStringRef,
    _parent: PhantomData<&'c Context>,
}

impl<'c> StringRef<'c> {
    pub fn from_str(context: &'c Context, string: &str) -> Self {
        let string = context
            .string_cache()
            .entry(CString::new(string).unwrap())
            .or_insert_with(Default::default)
            .key()
            .as_ptr();

        unsafe { Self::from_raw(mlirStringRefCreateFromCString(string)) }
    }

    /// Converts a string reference into a `str`.
    pub fn as_str(&self) -> Result<&'c str, Utf8Error> {
        unsafe {
            let bytes = slice::from_raw_parts(self.raw.data as *mut u8, self.raw.length);

            str::from_utf8(if bytes[bytes.len() - 1] == 0 {
                &bytes[..bytes.len() - 1]
            } else {
                bytes
            })
        }
    }

    /// Converts a string reference into a raw object.
    pub const fn to_raw(self) -> MlirStringRef {
        self.raw
    }

    /// Creates a string reference from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(string: MlirStringRef) -> Self {
        Self {
            raw: string,
            _parent: Default::default(),
        }
    }
}

impl<'a> PartialEq for StringRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirStringRefEqual(self.raw, other.raw) }
    }
}

impl<'a> Eq for StringRef<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal() {
        let context = Context::new();

        assert_eq!(
            StringRef::from_str(&context, "foo"),
            StringRef::from_str(&context, "foo")
        );
    }

    #[test]
    fn equal_str() {
        let context = Context::new();

        assert_eq!(
            StringRef::from_str(&context, "foo").as_str().unwrap(),
            "foo"
        );
    }

    #[test]
    fn not_equal() {
        let context = Context::new();

        assert_ne!(
            StringRef::from_str(&context, "foo"),
            StringRef::from_str(&context, "bar")
        );
    }
}
