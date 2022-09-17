//! General transformation passes.

use super::Pass;
use mlir_sys::{
    mlirCreateTransformsCSE, mlirCreateTransformsCanonicalizer, mlirCreateTransformsInliner,
    mlirCreateTransformsPrintOpStats, mlirCreateTransformsSCCP, mlirCreateTransformsStripDebugInfo,
    mlirCreateTransformsSymbolDCE, mlirCreateTransformsSymbolPrivatize,
};

/// Creates a pass to canonicalize IR.
pub fn canonicalizer() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsCanonicalizer)
}

/// Creates a pass to eliminate common sub-expressions.
pub fn cse() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsCSE)
}

/// Creates a pass to inline function calls.
pub fn inliner() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsInliner)
}

/// Creates a pass to propagate constants.
pub fn sccp() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsSCCP)
}

/// Creates a pass to strip debug information.
pub fn strip_debug_info() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsStripDebugInfo)
}

/// Creates a pass to eliminate dead symbols.
pub fn symbol_dce() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsSymbolDCE)
}

/// Creates a pass to mark all top-level symbols private.
pub fn symbol_privatize() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsSymbolPrivatize)
}

/// Creates a pass to print operation statistics.
pub fn print_operation_stats() -> Pass {
    Pass::from_raw_fn(mlirCreateTransformsPrintOpStats)
}
