use crate::{
    context::Context, dialect, logical_result::LogicalResult,
    operation_pass_manager::OperationPassManager, string_ref::StringRef,
};
use mlir_sys::{
    mlirParsePassPipeline, mlirRegisterAllDialects, mlirRegisterAllLLVMTranslations,
    mlirRegisterAllPasses, mlirRegisterTransformsCSE, mlirRegisterTransformsPrintOpStats,
};
use std::sync::Once;

/// Registers all dialects to a dialect registry.
pub fn register_all_dialects(registry: &dialect::Registry) {
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
pub fn parse_pass_pipeline(manager: OperationPassManager, source: &str) -> LogicalResult {
    LogicalResult::from_raw(unsafe {
        mlirParsePassPipeline(manager.to_raw(), StringRef::from(source).to_raw())
    })
}

/// Registers a pass to print operation stats.
pub fn register_print_operation_stats() {
    unsafe { mlirRegisterTransformsPrintOpStats() }
}

/// Registers a pass to print operation stats.
pub fn register_cse() {
    unsafe { mlirRegisterTransformsCSE() }
}

// TODO Use into_raw_parts.
pub(crate) unsafe fn into_raw_array<T>(xs: Vec<T>) -> *mut T {
    xs.leak().as_mut_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_dialects() {
        let registry = dialect::Registry::new();

        register_all_dialects(&registry);
    }

    #[test]
    fn register_dialects_twice() {
        let registry = dialect::Registry::new();

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
