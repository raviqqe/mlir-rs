//! Utility functions.

use crate::{
    context::Context, dialect::DialectRegistry, logical_result::LogicalResult, pass,
    string_ref::StringRef, Error,
};
use mlir_sys::{
    mlirParsePassPipeline, mlirRegisterAllDialects, mlirRegisterAllLLVMTranslations,
    mlirRegisterAllPasses, MlirStringRef,
};
use std::{
    ffi::c_void,
    fmt::{self, Formatter},
    sync::Once,
};

/// Registers all dialects to a dialect registry.
pub fn register_all_dialects(registry: &DialectRegistry) {
    unsafe { mlirRegisterAllDialects(registry.to_raw()) }
}

/// Register all translations from other dialects to the `llvm` dialect.
pub fn register_all_llvm_translations(context: &Context) {
    unsafe { mlirRegisterAllLLVMTranslations(context.to_raw()) }
}

/// Register all passes.
pub fn register_all_passes() {
    static ONCE: Once = Once::new();

    // Multiple calls of `mlirRegisterAllPasses` seems to cause double free.
    ONCE.call_once(|| unsafe { mlirRegisterAllPasses() });
}

/// Parses a pass pipeline.
pub fn parse_pass_pipeline(manager: pass::OperationManager, source: &str) -> Result<(), Error> {
    let mut error_message = None;

    let result = LogicalResult::from_raw(unsafe {
        mlirParsePassPipeline(
            manager.to_raw(),
            StringRef::from(source).to_raw(),
            Some(handle_parse_error),
            &mut error_message as *mut _ as *mut _,
        )
    });

    if result.is_success() {
        Ok(())
    } else {
        Err(Error::ParsePassPipeline(error_message.unwrap_or_else(
            || "failed to parse error message in UTF-8".into(),
        )))
    }
}

unsafe extern "C" fn handle_parse_error(raw_string: MlirStringRef, data: *mut c_void) {
    let string = StringRef::from_raw(raw_string);
    let data = &mut *(data as *mut Option<String>);

    if let Some(message) = data {
        message.extend(string.as_str())
    } else {
        *data = string.as_str().map(String::from).ok();
    }
}

// TODO Use into_raw_parts.
pub(crate) unsafe fn into_raw_array<T>(xs: Vec<T>) -> *mut T {
    xs.leak().as_mut_ptr()
}

pub(crate) unsafe extern "C" fn print_callback(string: MlirStringRef, data: *mut c_void) {
    let (formatter, result) = &mut *(data as *mut (&mut Formatter, fmt::Result));

    if result.is_err() {
        return;
    }

    *result = (|| -> fmt::Result {
        write!(
            formatter,
            "{}",
            StringRef::from_raw(string)
                .as_str()
                .map_err(|_| fmt::Error)?
        )
    })();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_dialects() {
        let registry = DialectRegistry::new();

        register_all_dialects(&registry);
    }

    #[test]
    fn register_dialects_twice() {
        let registry = DialectRegistry::new();

        register_all_dialects(&registry);
        register_all_dialects(&registry);
    }

    #[test]
    fn register_llvm_translations() {
        let context = Context::new();

        register_all_llvm_translations(&context);
    }

    #[test]
    fn register_llvm_translations_twice() {
        let context = Context::new();

        register_all_llvm_translations(&context);
        register_all_llvm_translations(&context);
    }

    #[test]
    fn register_passes() {
        register_all_passes();
    }

    #[test]
    fn register_passes_twice() {
        register_all_passes();
        register_all_passes();
    }

    #[test]
    fn register_passes_many_times() {
        for _ in 0..1000 {
            register_all_passes();
        }
    }
}
