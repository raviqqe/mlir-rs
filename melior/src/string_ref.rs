use crate::Context;
use mlir_sys::{mlirStringRefEqual, MlirStringRef};
use std::{
    ffi::CStr,
    marker::PhantomData,
    pin::Pin,
    slice,
    str::{self, Utf8Error},
};

/// A string reference.
// https://mlir.llvm.org/docs/CAPI/#stringref
//
// TODO The documentation says string refs do not have to be null-terminated.
// But it looks like some functions do not handle strings not null-terminated?
#[derive(Clone, Copy, Debug)]
pub struct StringRef<'a> {
    raw: MlirStringRef,
    _parent: PhantomData<&'a str>,
}

impl<'a> StringRef<'a> {
    /// Creates a string reference.
    pub fn new(string: &'a str) -> Self {
        let string = MlirStringRef {
            data: string.as_bytes().as_ptr() as *const i8,
            length: string.len(),
        };

        unsafe { Self::from_raw(string) }
    }

    /// Converts a C-style string into a string reference.
    pub fn from_c_str(string: &'a CStr) -> Self {
        let string = MlirStringRef {
            data: string.as_ptr(),
            length: string.to_bytes_with_nul().len() - 1,
        };

        unsafe { Self::from_raw(string) }
    }

    /// Converts a string into a null-terminated string reference.
    #[deprecated]
    pub fn from_str(context: &'a Context, string: &str) -> Self {
        let entry = context
            .string_cache()
            .entry(Pin::new(string.into()))
            .or_default();
        let string = MlirStringRef {
            data: entry.key().as_bytes().as_ptr() as *const i8,
            length: entry.key().len(),
        };

        unsafe { Self::from_raw(string) }
    }

    /// Converts a string reference into a `str`.
    pub fn as_str(&self) -> Result<&'a str, Utf8Error> {
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
