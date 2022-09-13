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
        self.result.value != 0
    }

    pub fn is_failure(&self) -> bool {
        self.result.value == 0
    }

    pub(crate) fn from_raw(result: MlirLogicalResult) -> Self {
        Self { result }
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
