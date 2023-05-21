use mlir_sys::{mlirOpPrintingFlagsCreate, MlirOpPrintingFlags};

#[derive(Clone, Copy, Debug)]
pub struct OperationPrintingFlags(MlirOpPrintingFlags);

impl OperationPrintingFlags {
    pub fn new() -> Self {
        Self(unsafe { mlirOpPrintingFlagsCreate() })
    }
}
