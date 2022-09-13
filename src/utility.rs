use crate::{context::Context, dialect_registry::DialectRegistry};
use mlir_sys::{mlirRegisterAllDialects, mlirRegisterAllLLVMTranslations};

/// Registers all dialects to a dialect registry.
pub fn register_all_dialects(registry: &DialectRegistry) {
    unsafe { mlirRegisterAllDialects(registry.to_raw()) }
}

/// Register all translations from other dialects to the `llvm` dialect.
pub fn register_all_llvm_translations(context: &Context) {
    unsafe { mlirRegisterAllLLVMTranslations(context.to_raw()) }
}

// TODO Use into_raw_parts.
pub(crate) unsafe fn into_raw_array<T>(xs: Vec<T>) -> *mut T {
    xs.leak().as_mut_ptr()
}
