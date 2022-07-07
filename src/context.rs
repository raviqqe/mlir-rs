#[derive(Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Context::new();
    }
}
