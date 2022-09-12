use mlir_sys::MlirLogicalResult;

pub struct LogicalResult {
    result: MlirLogicalResult,
}

impl LogicalResult {
    pub fn success() -> Self {
        Self {
            result: MlirLogicalResult { value: 1 },
        }
    }

    pub fn failure() -> Self {
        Self {
            result: MlirLogicalResult { value: 0 },
        }
    }

    pub fn is_success(&self) -> bool {
        unsafe { self.result.value != 0 }
    }

    pub fn is_failure(&self) -> bool {
        unsafe { self.result.value == 0 }
    }

    pub(crate) unsafe fn from_raw(result: MlirLogicalResult) -> Self {
        Self { result }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirLogicalResult {
        self.result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success() {
        assert!(LogicalResult::success().is_success());
    }

    #[test]
    fn failure() {
        assert!(LogicalResult::failure().is_failure());
    }
}
