#[derive(Debug)]
pub struct Context {
    context: mlir_sys::MlirContext,
}

impl Context {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            context: unsafe { mlir_sys::mlirContextCreate() },
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { mlir_sys::mlirContextDestroy(self.context) };
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
