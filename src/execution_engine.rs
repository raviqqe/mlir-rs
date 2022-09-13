use crate::{logical_result::LogicalResult, module::Module, string_ref::StringRef};
use mlir_sys::{
    mlirExecutionEngineCreate, mlirExecutionEngineDestroy, mlirExecutionEngineInvokePacked,
    MlirExecutionEngine,
};
use std::ffi::c_void;

pub struct ExecutionEngine {
    raw: MlirExecutionEngine,
}

impl ExecutionEngine {
    pub fn new(module: &Module, optimization_level: usize, shared_library_paths: &[&str]) -> Self {
        Self {
            raw: unsafe {
                mlirExecutionEngineCreate(
                    module.to_raw(),
                    optimization_level as i32,
                    shared_library_paths.len() as i32,
                    shared_library_paths
                        .iter()
                        .map(|&string| StringRef::from(string).to_raw())
                        .collect::<Vec<_>>()
                        .as_ptr(),
                )
            },
        }
    }

    // # Safety
    //
    // This function modifies memory locations pointed by the `arguments` argument.
    // If those pointers are invalid or misaligned, calling this function might result in undefined
    // behavior.
    pub unsafe fn invoke_packed(&self, name: &str, arguments: &mut [*mut ()]) -> LogicalResult {
        LogicalResult::from_raw(mlirExecutionEngineInvokePacked(
            self.raw,
            StringRef::from(name).to_raw(),
            arguments.as_mut_ptr() as *mut *mut c_void,
        ))
    }
}

impl Drop for ExecutionEngine {
    fn drop(&mut self) {
        unsafe { mlirExecutionEngineDestroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context, dialect_registry::DialectRegistry, module::Module, pass::Pass,
        pass_manager::PassManager,
    };

    #[test]
    fn invoke_packed() {
        let registry = DialectRegistry::new();
        registry.register_all_dialects();

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.register_all_llvm_translations();

        let mut module = Module::parse(
            &context,
            r#"
            module {
                func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
            }
            "#,
        );

        let mut pass_manager = PassManager::new(&context);
        pass_manager.add_pass(Pass::convert_func_to_llvm());

        pass_manager
            .nested_under("func.func")
            .add_pass(Pass::convert_arithmetic_to_llvm());

        assert!(pass_manager.run(&mut module).is_success());

        let engine = ExecutionEngine::new(&module, 2, &[]);

        let mut argument = 42;
        let mut result = -1;

        assert!(unsafe {
            engine.invoke_packed(
                "add",
                &mut [
                    &mut argument as *mut i32 as *mut (),
                    &mut result as *mut i32 as *mut (),
                ],
            )
        }
        .is_success());

        assert_eq!(argument, 42);
        assert_eq!(result, 84);
    }
}
