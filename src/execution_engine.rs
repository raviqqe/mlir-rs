use crate::{logical_result::LogicalResult, module::Module, string_ref::StringRef};
use mlir_sys::{mlirExecutionEngineCreate, mlirExecutionEngineInvokePacked, MlirExecutionEngine};
use std::ffi::c_void;

pub struct ExecutionEngine {
    engine: MlirExecutionEngine,
}

impl ExecutionEngine {
    pub fn new(module: &Module, optimization_level: usize, shared_library_paths: &[&str]) -> Self {
        Self {
            engine: unsafe {
                mlirExecutionEngineCreate(
                    module.to_raw(),
                    optimization_level as i32,
                    shared_library_paths.len() as i32,
                    sharedLibPaths,
                )
            },
        }
    }

    pub unsafe fn invoke_packed(&self, name: &str, arguments: &mut [*mut ()]) -> LogicalResult {
        unsafe {
            LogicalResult::from_raw(mlirExecutionEngineInvokePacked(
                self.engine,
                StringRef::from(name).to_raw(),
                arguments.as_mut_ptr() as *mut *mut c_void,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::Context, dialect_registry::DialectRegistry, module::Module};

    #[test]
    fn simple() {
        let registry = DialectRegistry::new();
        let context = Context::new();

        let module = Module::parse(
            &context,
            r#"
      module {                                                                    \n
        func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {     \n
          %res = arith.addi %arg0, %arg0 : i32                                        \n
          return %res : i32                                                           \n
        }                                                                             \n
      }
      "#,
        );

        // TODO
        // lowerModuleToLLVM(ctx, module);
        context.register_all_llvm_translations();
        let engine = ExecutionEngine::new(&module, 2, &[]);

        let input = 42;
        let result = -1;
        let args = vec![&mut input, &mut result];

        assert!(unsafe {
            engine.invoke_packed(
                "add",
                &mut [
                    &mut input as *mut i32 as *mut (),
                    &mut result as *mut i32 as *mut (),
                ],
            )
        }
        .is_success());
    }
}
