use mlir_sys::MlirDiagnosticHandlerID;

/// Diagnostic handler ID.
#[derive(Clone, Copy, Debug)]
pub struct DiagnosticHandlerId {
    raw: MlirDiagnosticHandlerID,
}

impl DiagnosticHandlerId {
    /// Creates a diagnostic handler ID from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub const unsafe fn from_raw(raw: MlirDiagnosticHandlerID) -> Self {
        Self { raw }
    }

    /// Converts a diagnostic handler ID into a raw object.
    pub const fn to_raw(self) -> MlirDiagnosticHandlerID {
        self.raw
    }
}
