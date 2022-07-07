use crate::{context::Context, location::Location};
use std::marker::PhantomData;

pub struct Module<'c> {
    module: mlir_sys::MlirModule,
    _context: PhantomData<&'c Context>,
}

impl<'c> Module<'c> {
    pub fn new(location: Location) -> Self {
        Self {
            module: unsafe { mlir_sys::mlirModuleCreateEmpty(location.to_raw()) },
            _context: Default::default(),
        }
    }

    pub unsafe fn to_raw(&self) -> mlir_sys::MlirModule {
        self.module
    }
}

impl<'c> Drop for Module<'c> {
    fn drop(&mut self) {
        unsafe { mlir_sys::mlirModuleDestroy(self.module) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42));
    }
}
