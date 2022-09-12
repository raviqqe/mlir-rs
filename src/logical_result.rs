use mlir_sys::MlirLogicalResult;

pub struct LogicalResult {
    result: MlirLogicalResult,
}

impl LogicalResult {
    pub(crate) unsafe fn from_raw(result: MlirLogicalResult) -> Self {
        Self { result }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirLogicalResult {
        self.result
    }
}
