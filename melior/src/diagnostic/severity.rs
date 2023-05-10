use crate::Error;
use mlir_sys::{
    MlirDiagnosticSeverity_MlirDiagnosticError, MlirDiagnosticSeverity_MlirDiagnosticNote,
    MlirDiagnosticSeverity_MlirDiagnosticRemark, MlirDiagnosticSeverity_MlirDiagnosticWarning,
};

/// Diagnostic severity.
#[derive(Clone, Copy, Debug)]
pub enum DiagnosticSeverity {
    Error,
    Note,
    Remark,
    Warning,
}

impl TryFrom<u32> for DiagnosticSeverity {
    type Error = Error;

    fn try_from(severity: u32) -> Result<Self, Error> {
        #[allow(non_upper_case_globals)]
        Ok(match severity {
            MlirDiagnosticSeverity_MlirDiagnosticError => Self::Error,
            MlirDiagnosticSeverity_MlirDiagnosticNote => Self::Note,
            MlirDiagnosticSeverity_MlirDiagnosticRemark => Self::Remark,
            MlirDiagnosticSeverity_MlirDiagnosticWarning => Self::Warning,
            _ => return Err(Error::UnknownDiagnosticSeverity(severity)),
        })
    }
}
