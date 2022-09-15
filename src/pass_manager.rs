use crate::{
    context::Context, logical_result::LogicalResult, module::Module,
    operation_pass_manager::OperationPassManager, pass::Pass, string_ref::StringRef,
};
use mlir_sys::{
    mlirPassManagerAddOwnedPass, mlirPassManagerCreate, mlirPassManagerDestroy,
    mlirPassManagerEnableIRPrinting, mlirPassManagerEnableVerifier,
    mlirPassManagerGetAsOpPassManager, mlirPassManagerGetNestedUnder, mlirPassManagerRun,
    MlirPassManager,
};
use std::marker::PhantomData;

/// A pass manager.
pub struct PassManager<'c> {
    raw: MlirPassManager,
    _context: PhantomData<&'c Context>,
}

impl<'c> PassManager<'c> {
    /// Creates a pass manager.
    pub fn new(context: &Context) -> Self {
        Self {
            raw: unsafe { mlirPassManagerCreate(context.to_raw()) },
            _context: Default::default(),
        }
    }

    /// Gets an operation pass manager for nested operations corresponding to a
    /// given name.
    pub fn nested_under(&self, name: &str) -> OperationPassManager {
        unsafe {
            OperationPassManager::from_raw(mlirPassManagerGetNestedUnder(
                self.raw,
                StringRef::from(name).to_raw(),
            ))
        }
    }

    /// Adds a pass.
    pub fn add_pass(&self, pass: Pass) {
        unsafe { mlirPassManagerAddOwnedPass(self.raw, pass.to_raw()) }
    }

    /// Enables a verifier.
    pub fn enable_verifier(&self, enabled: bool) {
        unsafe { mlirPassManagerEnableVerifier(self.raw, enabled) }
    }

    /// Enables IR printing.
    pub fn enable_ir_printing(&self) {
        unsafe { mlirPassManagerEnableIRPrinting(self.raw) }
    }

    /// Runs passes added to a pass manager against a module.
    pub fn run(&self, module: &Module) -> LogicalResult {
        LogicalResult::from_raw(unsafe { mlirPassManagerRun(self.raw, module.to_raw()) })
    }

    /// Converts a pass manager to an operation pass manager.
    pub fn as_operation_pass_manager(&self) -> OperationPassManager {
        unsafe { OperationPassManager::from_raw(mlirPassManagerGetAsOpPassManager(self.raw)) }
    }
}

impl<'c> Drop for PassManager<'c> {
    fn drop(&mut self) {
        unsafe { mlirPassManagerDestroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect_registry::DialectRegistry,
        location::Location,
        utility::{parse_pass_pipeline, register_all_dialects, register_print_operation_stats},
    };
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    fn register_all_upstream_dialects(context: &Context) {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
    }

    #[test]
    fn new() {
        let context = Context::new();

        PassManager::new(&context);
    }

    #[test]
    fn add_pass() {
        let context = Context::new();

        PassManager::new(&context).add_pass(Pass::convert_func_to_llvm());
    }

    #[test]
    fn enable_verifier() {
        let context = Context::new();

        PassManager::new(&context).enable_verifier(true);
    }

    // TODO Enable this test.
    // #[test]
    // fn enable_ir_printing() {
    //     let context = Context::new();

    //     PassManager::new(&context).enable_ir_printing();
    // }

    #[test]
    fn run() {
        let context = Context::new();
        let manager = PassManager::new(&context);

        manager.add_pass(Pass::convert_func_to_llvm());
        manager.run(&Module::new(Location::unknown(&context)));
    }

    #[test]
    fn run_on_function() {
        let context = Context::new();
        register_all_upstream_dialects(&context);

        let module = Module::parse(
            &context,
            indoc!(
                "
                func.func @foo(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
                "
            ),
        )
        .unwrap();

        let manager = PassManager::new(&context);
        manager.add_pass(Pass::print_operation_stats());

        assert!(manager.run(&module).is_success());
    }

    #[test]
    fn run_on_function_in_nested_module() {
        let context = Context::new();
        register_all_upstream_dialects(&context);

        let module = Module::parse(
            &context,
            indoc!(
                "
                func.func @foo(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }

                module {
                    func.func @bar(%arg0 : f32) -> f32 {
                        %res = arith.addf %arg0, %arg0 : f32
                        return %res : f32
                    }
                }
                "
            ),
        )
        .unwrap();

        let manager = PassManager::new(&context);
        manager
            .nested_under("func.func")
            .add_pass(Pass::print_operation_stats());

        assert!(manager.run(&module).is_success());

        let manager = PassManager::new(&context);
        manager
            .nested_under("builtin.module")
            .nested_under("func.func")
            .add_pass(Pass::print_operation_stats());

        assert!(manager.run(&module).is_success());
    }

    #[test]
    fn print_pass_pipeline() {
        let context = Context::new();
        let manager = PassManager::new(&context);
        let module_manager = manager.nested_under("builtin.module");
        let function_manager = module_manager.nested_under("func.func");

        function_manager.add_pass(Pass::print_operation_stats());

        assert_eq!(
            manager.as_operation_pass_manager().to_string(),
            "builtin.module(func.func(print-op-stats{json=false}))"
        );
        assert_eq!(
            module_manager.to_string(),
            "func.func(print-op-stats{json=false})"
        );
        assert_eq!(function_manager.to_string(), "print-op-stats{json=false}");
    }

    #[test]
    fn parse_pass_pipeline_() {
        let context = Context::new();
        let manager = PassManager::new(&context);

        assert!(parse_pass_pipeline(
            manager.as_operation_pass_manager(),
            "builtin.module(func.func(print-op-stats{json=false}),\
                func.func(print-op-stats{json=false}))"
        )
        .is_failure());

        register_print_operation_stats();

        assert!(parse_pass_pipeline(
            manager.as_operation_pass_manager(),
            "builtin.module(func.func(print-op-stats{json=false}),\
                func.func(print-op-stats{json=false}))"
        )
        .is_success());

        assert_eq!(
            manager.as_operation_pass_manager().to_string(),
            "builtin.module(func.func(print-op-stats{json=false}),\
            func.func(print-op-stats{json=false}))"
        );
    }
}
