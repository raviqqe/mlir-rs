use crate::{context::Context, utility};
use std::marker::PhantomData;

pub struct Location<'c> {
    location: mlir_sys::MlirLocation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Location<'c> {
    pub fn new(context: &Context, filename: &str, line: usize, column: usize) -> Self {
        Self {
            location: unsafe {
                mlir_sys::mlirLocationFileLineColGet(
                    context.to_raw(),
                    utility::as_string_ref(filename),
                    line as u32,
                    column as u32,
                )
            },
            _context: Default::default(),
        }
    }

    pub unsafe fn to_raw(&self) -> mlir_sys::MlirLocation {
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
}
