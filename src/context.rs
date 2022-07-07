use std::{marker::PhantomData, mem::ManuallyDrop, ops::Deref};

pub struct Context {
    context: mlir_sys::MlirContext,
}

impl Context {
    pub fn new() -> Self {
        Self {
            context: unsafe { mlir_sys::mlirContextCreate() },
        }
    }

    pub unsafe fn to_raw(&self) -> mlir_sys::MlirContext {
        self.context
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { mlir_sys::mlirContextDestroy(self.context) };
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ContextRef<'c> {
    context: ManuallyDrop<Context>,
    _context: PhantomData<&'c Context>,
}

impl<'c> ContextRef<'c> {
    pub unsafe fn from_raw(context: mlir_sys::MlirContext) -> Self {
        Self {
            context: ManuallyDrop::new(Context { context }),
            _context: Default::default(),
        }
    }
}

impl<'c> Deref for ContextRef<'c> {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &*self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Context::new();
    }
}
