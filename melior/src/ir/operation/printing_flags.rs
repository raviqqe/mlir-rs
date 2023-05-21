use mlir_sys::{mlirOpPrintingFlagsCreate, mlirOpPrintingFlagsDestroy, MlirOpPrintingFlags};

#[derive(Debug)]
pub struct OperationPrintingFlags(MlirOpPrintingFlags);

impl OperationPrintingFlags {
    pub fn new() -> Self {
        Self(unsafe { mlirOpPrintingFlagsCreate() })
    }

    pub fn foo() -> foo {}
}

impl Drop for OperationPrintingFlags {
    fn drop(&mut self) {
        unsafe { mlirOpPrintingFlagsDestroy(self.0) }
    }
}
