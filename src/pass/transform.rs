//! General transformation passes.

use super::Pass;
use mlir_sys::{
    mlirCreateTransformsCSE, mlirCreateTransformsCanonicalizer, mlirCreateTransformsInliner,
    mlirCreateTransformsPrintOpStats, mlirCreateTransformsSCCP, mlirCreateTransformsStripDebugInfo,
    mlirCreateTransformsSymbolDCE, mlirCreateTransformsSymbolPrivatize, mlirRegisterTransformsCSE,
    mlirRegisterTransformsCanonicalizer, mlirRegisterTransformsInliner,
    mlirRegisterTransformsPrintOpStats, mlirRegisterTransformsSCCP,
    mlirRegisterTransformsStripDebugInfo, mlirRegisterTransformsSymbolDCE,
    mlirRegisterTransformsSymbolPrivatize,
};

/// Creates a pass to canonicalize IR.
pub fn canonicalizer() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsCanonicalizer)
}

/// Registers a pass to canonicalize IR.
pub fn register_canonicalizer() {
    unsafe { mlirRegisterTransformsCanonicalizer() }
}

/// Creates a pass to eliminate common sub-expressions.
pub fn cse() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsCSE)
}

/// Registers a pass to print operation stats.
pub fn register_cse() {
    unsafe { mlirRegisterTransformsCSE() }
}

/// Creates a pass to inline function calls.
pub fn inliner() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsInliner)
}

/// Registers a pass to inline function calls.
pub fn register_inliner() {
    unsafe { mlirRegisterTransformsInliner() }
}

/// Creates a pass to propagate constants.
pub fn sccp() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsSCCP)
}

/// Registers a pass to propagate constants.
pub fn register_sccp() {
    unsafe { mlirRegisterTransformsSCCP() }
}

/// Creates a pass to strip debug information.
pub fn strip_debug_info() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsStripDebugInfo)
}

/// Registers a pass to strip debug information.
pub fn register_strip_debug_info() {
    unsafe { mlirRegisterTransformsStripDebugInfo() }
}

/// Creates a pass to eliminate dead symbols.
pub fn symbol_dce() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsSymbolDCE)
}

/// Registers a pass to eliminate dead symbols.
pub fn register_symbol_dce() {
    unsafe { mlirRegisterTransformsSymbolDCE() }
}

/// Creates a pass to mark all top-level symbols private.
pub fn symbol_privatize() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsSymbolPrivatize)
}

/// Registers a pass to mark all top-level symbols private.
pub fn register_symbol_privatize() {
    unsafe { mlirRegisterTransformsSymbolPrivatize() }
}

/// Creates a pass to print operation statistics.
pub fn print_operation_stats() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsPrintOpStats)
}

/// Registers a pass to print operation stats.
pub fn register_print_operation_stats() {
    unsafe { mlirRegisterTransformsPrintOpStats() }
}
