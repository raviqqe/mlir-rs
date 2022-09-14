use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{
    mlirLocationEqual, mlirLocationFileLineColGet, mlirLocationGetContext, mlirLocationUnknownGet,
    MlirLocation,
};
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug)]
pub struct Location<'c> {
    raw: MlirLocation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Location<'c> {
    pub fn new(context: &'c Context, filename: &str, line: usize, column: usize) -> Self {
        Self {
            raw: unsafe {
                mlirLocationFileLineColGet(
                    context.to_raw(),
                    StringRef::from(filename).to_raw(),
                    line as u32,
                    column as u32,
                )
            },
            _context: Default::default(),
        }
    }

    pub fn unknown(context: &Context) -> Self {
        Self {
            raw: unsafe { mlirLocationUnknownGet(context.to_raw()) },
            _context: Default::default(),
        }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirLocationGetContext(self.raw)) }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
