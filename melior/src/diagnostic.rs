use crate::{ir::Location, logical_result::LogicalResult, utility::print_callback, Context};
use mlir_sys::{
    mlirContextAttachDiagnosticHandler, mlirContextDetachDiagnosticHandler,
    mlirDiagnosticGetLocation, mlirDiagnosticGetNote, mlirDiagnosticGetNumNotes,
    mlirDiagnosticGetSeverity, mlirDiagnosticPrint, MlirDiagnostic, MlirDiagnosticHandlerID,
    MlirDiagnosticSeverity_MlirDiagnosticError, MlirDiagnosticSeverity_MlirDiagnosticNote,
    MlirDiagnosticSeverity_MlirDiagnosticRemark, MlirDiagnosticSeverity_MlirDiagnosticWarning,
    MlirLogicalResult,
};
use std::{ffi::c_void, fmt, marker::PhantomData};

#[derive(Debug)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Note,
    Remark,
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
        #[allow(non_upper_case_globals)]
        match unsafe { mlirDiagnosticGetSeverity(self.raw) } {
            MlirDiagnosticSeverity_MlirDiagnosticError => DiagnosticSeverity::Error,
            MlirDiagnosticSeverity_MlirDiagnosticWarning => DiagnosticSeverity::Warning,
            MlirDiagnosticSeverity_MlirDiagnosticNote => DiagnosticSeverity::Note,
            MlirDiagnosticSeverity_MlirDiagnosticRemark => DiagnosticSeverity::Remark,
            _ => unreachable!(),
        }
    }

    pub fn note_count(&self) -> usize {
        unsafe { mlirDiagnosticGetNumNotes(self.raw) as usize }
    }

    pub fn note(&self, index: usize) -> Self {
        unsafe { Self::from_raw(mlirDiagnosticGetNote(self.raw, index as isize)) }
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

#[derive(Debug)]
pub struct DiagnosticHandler {
    raw: MlirDiagnosticHandlerID,
}

impl Context {
    /// Attach a diagnostic handler to the context.
    pub fn attach_diagnostic_handler<F>(&self, handler: F) -> DiagnosticHandler
    where
        F: FnMut(Diagnostic) -> LogicalResult,
    {
        let handler = Box::new(handler);
        let handler = Box::into_raw(handler);

        let handler = unsafe {
            mlirContextAttachDiagnosticHandler(
                self.to_raw(),
                Some(_mlir_cb_invoke::<F>),
                handler as *mut c_void,
                Some(_mlir_cb_detach::<F>),
            )
        };

        DiagnosticHandler { raw: handler }
    }

    pub fn detach_diagnostic_handler<F>(&self, handler: DiagnosticHandler) {
        unsafe {
            mlirContextDetachDiagnosticHandler(self.to_raw(), handler.raw);
        }
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
