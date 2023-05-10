use crate::{ir::Location, utility::print_callback, Error};
use mlir_sys::{
    mlirDiagnosticGetLocation, mlirDiagnosticGetNote, mlirDiagnosticGetNumNotes,
    mlirDiagnosticGetSeverity, mlirDiagnosticPrint, MlirDiagnostic, MlirDiagnosticHandlerID,
    MlirDiagnosticSeverity_MlirDiagnosticError, MlirDiagnosticSeverity_MlirDiagnosticNote,
    MlirDiagnosticSeverity_MlirDiagnosticRemark, MlirDiagnosticSeverity_MlirDiagnosticWarning,
};
use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
};

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
        #[allow(non_upper_case_globals)]
        match unsafe { mlirDiagnosticGetSeverity(self.raw) } {
            MlirDiagnosticSeverity_MlirDiagnosticError => DiagnosticSeverity::Error,
            MlirDiagnosticSeverity_MlirDiagnosticNote => DiagnosticSeverity::Note,
            MlirDiagnosticSeverity_MlirDiagnosticRemark => DiagnosticSeverity::Remark,
            MlirDiagnosticSeverity_MlirDiagnosticWarning => DiagnosticSeverity::Warning,
            _ => unreachable!("unexpected diagnostic severity"),
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

impl<'a> Display for Diagnostic<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

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
    pub(crate) unsafe fn from_raw(raw: MlirDiagnosticHandlerID) -> Self {
        Self { raw }
    }

    pub(crate) unsafe fn to_raw(self) -> MlirDiagnosticHandlerID {
        self.raw
    }
}
