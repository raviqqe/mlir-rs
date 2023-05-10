use crate::{ir::Location, logical_result::LogicalResult, utility::print_callback, Error};
use mlir_sys::{
    mlirDiagnosticGetLocation, mlirDiagnosticGetNote, mlirDiagnosticGetNumNotes,
    mlirDiagnosticGetSeverity, mlirDiagnosticPrint, MlirDiagnostic, MlirDiagnosticHandlerID,
    MlirDiagnosticSeverity_MlirDiagnosticError, MlirDiagnosticSeverity_MlirDiagnosticNote,
    MlirDiagnosticSeverity_MlirDiagnosticRemark, MlirDiagnosticSeverity_MlirDiagnosticWarning,
    MlirLogicalResult,
};
use std::{ffi::c_void, fmt, marker::PhantomData};

#[derive(Clone, Copy, Debug)]
pub enum DiagnosticSeverity {
    Error,
    Note,
    Remark,
    Warning,
}

#[derive(Debug)]
pub struct Diagnostic<'a> {
    raw: MlirDiagnostic,
    phantom: PhantomData<&'a ()>,
}

impl<'a> Diagnostic<'a> {
    pub fn location(&self) -> Location {
        unsafe { Location::from_raw(mlirDiagnosticGetLocation(self.raw)) }
    }

    pub fn severity(&self) -> DiagnosticSeverity {
        match unsafe { mlirDiagnosticGetSeverity(self.raw) } {
            MlirDiagnosticSeverity_MlirDiagnosticError => DiagnosticSeverity::Error,
            MlirDiagnosticSeverity_MlirDiagnosticNote => DiagnosticSeverity::Note,
            MlirDiagnosticSeverity_MlirDiagnosticRemark => DiagnosticSeverity::Remark,
            MlirDiagnosticSeverity_MlirDiagnosticWarning => DiagnosticSeverity::Warning,
        }
    }

    pub fn note_count(&self) -> usize {
        (unsafe { mlirDiagnosticGetNumNotes(self.raw) }) as usize
    }

    pub fn note(&self, index: usize) -> Result<Self, Error> {
        if index < self.note_count() {
            Ok(unsafe { Self::from_raw(mlirDiagnosticGetNote(self.raw, index as isize)) })
        } else {
            Err(Error::PositionOutOfBounds(
                "diagnostic note",
                self.to_string(),
                index,
            ))
        }
    }

    pub(crate) unsafe fn from_raw(raw: MlirDiagnostic) -> Self {
        Self {
            raw,
            phantom: Default::default(),
        }
    }
}

impl<'a> fmt::Display for Diagnostic<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut data = (f, Ok(()));

        unsafe {
            mlirDiagnosticPrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DiagnosticHandlerId {
    raw: MlirDiagnosticHandlerID,
}

impl DiagnosticHandlerId {
    pub(crate) unsafe fn to_raw(&self) -> MlirDiagnosticHandlerID {
        self.raw
    }
}

unsafe extern "C" fn _mlir_cb_invoke<F>(
    diagnostic: MlirDiagnostic,
    user_data: *mut c_void,
) -> MlirLogicalResult
where
    F: FnMut(Diagnostic) -> LogicalResult,
{
    let diagnostic = Diagnostic::from_raw(diagnostic);

    let handler: &mut F = &mut *(user_data as *mut F);
    handler(diagnostic).to_raw()
}

unsafe extern "C" fn _mlir_cb_detach<F>(user_data: *mut c_void)
where
    F: FnMut(Diagnostic) -> LogicalResult,
{
    drop(Box::from_raw(user_data as *mut F));
}
