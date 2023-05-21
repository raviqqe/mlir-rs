use mlir_sys::{
    mlirOpPrintingFlagsCreate, mlirOpPrintingFlagsDestroy,
    mlirOpPrintingFlagsElideLargeElementsAttrs, MlirOpPrintingFlags,
};

#[derive(Debug)]
pub struct OperationPrintingFlags(MlirOpPrintingFlags);

impl OperationPrintingFlags {
    pub fn new() -> Self {
        Self(unsafe { mlirOpPrintingFlagsCreate() })
    }

    pub fn elide_large_elements_attributes(self, limit: usize) -> Self {
        unsafe { mlirOpPrintingFlagsElideLargeElementsAttrs(self.0, limit) }
        self
    }
}

impl Drop for OperationPrintingFlags {
    fn drop(&mut self) {
        unsafe { mlirOpPrintingFlagsDestroy(self.0) }
    }
}
