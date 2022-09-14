use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{
    mlirLocationEqual, mlirLocationFileLineColGet, mlirLocationGetContext, mlirLocationPrint,
    mlirLocationUnknownGet, MlirLocation, MlirStringRef,
};
use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
};

#[derive(Clone, Copy, Debug)]
pub struct Location<'c> {
    raw: MlirLocation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Location<'c> {
    pub fn new(context: &'c Context, filename: &str, line: usize, column: usize) -> Self {
        unsafe {
            Self::from_raw(mlirLocationFileLineColGet(
                context.to_raw(),
                StringRef::from(filename).to_raw(),
                line as u32,
                column as u32,
            ))
        }
    }

    pub fn unknown(context: &Context) -> Self {
        unsafe { Self::from_raw(mlirLocationUnknownGet(context.to_raw())) }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirLocationGetContext(self.raw)) }
    }

    pub(crate) unsafe fn from_raw(raw: MlirLocation) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    pub(crate) unsafe fn to_raw(self) -> MlirLocation {
        self.raw
    }
}

impl<'c> PartialEq for Location<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirLocationEqual(self.raw, other.raw) }
    }
}

impl<'c> Display for Location<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe extern "C" fn callback(string: MlirStringRef, data: *mut c_void) {
            let data = &mut *(data as *mut (&mut Formatter, fmt::Result));
            let result = write!(data.0, "{}", StringRef::from_raw(string).as_str());

            if data.1.is_ok() {
                data.1 = result;
            }
        }

        unsafe {
            mlirLocationPrint(self.raw, Some(callback), &mut data as *mut _ as *mut c_void);
        }

        data.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};

    #[test]
    fn new() {
        Location::new(&Context::new(), "foo", 42, 42);
    }

    #[test]
    fn unknown() {
        Location::unknown(&Context::new());
    }

    #[test]
    fn context() {
        Location::new(&Context::new(), "foo", 42, 42).context();
    }

    #[test]
    fn equal() {
        let context = Context::new();

        assert_eq!(Location::unknown(&context), Location::unknown(&context));
        assert_eq!(
            Location::new(&context, "foo", 42, 42),
            Location::new(&context, "foo", 42, 42),
        );
    }

    #[test]
    fn not_equal() {
        let context = Context::new();

        assert_ne!(
            Location::new(&context, "foo", 42, 42),
            Location::unknown(&context)
        );
    }

    #[test]
    fn display() {
        let context = Context::new();

        assert_eq!(Location::unknown(&context).to_string(), "loc(unknown)");
        assert_eq!(
            Location::new(&context, "foo", 42, 42).to_string(),
            "loc(\"foo\":42:42)"
        );
    }
}
