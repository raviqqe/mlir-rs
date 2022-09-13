use mlir_sys::MlirLogicalResult;

/// A logical result of success or failure.
pub struct LogicalResult {
    raw: MlirLogicalResult,
}

impl LogicalResult {
    pub fn success() -> Self {
        Self {
            raw: MlirLogicalResult { value: 1 },
        }
    }

    pub fn failure() -> Self {
        Self {
            raw: MlirLogicalResult { value: 0 },
        }
    }

    pub fn is_success(&self) -> bool {
        self.raw.value != 0
    }

    pub fn is_failure(&self) -> bool {
        self.raw.value == 0
    }

    pub(crate) fn from_raw(result: MlirLogicalResult) -> Self {
        Self { raw: result }
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
