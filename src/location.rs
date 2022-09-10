use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{
    mlirLocationFileLineColGet, mlirLocationGetContext, mlirLocationUnknownGet, MlirLocation,
};
use std::marker::PhantomData;

pub struct Location<'c> {
    location: MlirLocation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Location<'c> {
    pub fn new(context: &Context, filename: &str, line: usize, column: usize) -> Self {
        Self {
            location: unsafe {
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
            location: unsafe { mlirLocationUnknownGet(context.to_raw()) },
            _context: Default::default(),
        }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirLocationGetContext(self.location)) }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirLocation {
        self.location
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
}
