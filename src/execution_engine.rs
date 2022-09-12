use crate::{module::Module, string_ref::StringRef};
use mlir_sys::{mlirExecutionEngineCreate, mlirExecutionEngineInvokePacked, MlirExecutionEngine};

pub struct ExecutionEngine {
    engine: MlirExecutionEngine,
}

impl ExecutionEngine {
    pub fn new(
        module: &Module,
        optimization_level: usize,
        shared_library_paths: Vec<&str>,
    ) -> Self {
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

    pub unsafe fn invoke_packed(&self, name: &str, argments: &mut [&mut ()]) -> () {
        unsafe {
            mlirExecutionEngineInvokePacked(
                self.engine,
                StringRef::from(name).to_raw(),
                arguments.as_ptr(),
            )
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
        let engine = ExecutionEngine::new(&module, 2, vec![]);

        let input = 42;
        let result = -1;
        let args = vec![&input, &result];

        assert!();
        if (mlirLogicalResultIsFailure(mlirExecutionEngineInvokePacked(
            jit,
            mlirStringRefCreateFromCString("add"),
            args,
        ))) {
            fprintf(stderr, "Execution engine creation failed");
            abort();
        }
    }
}
