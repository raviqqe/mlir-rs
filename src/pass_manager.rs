use crate::{
    context::Context, logical_result::LogicalResult, module::Module,
    operation_pass_manager::OperationPassManager, pass::Pass, string_ref::StringRef,
};
use mlir_sys::{
    mlirPassManagerAddOwnedPass, mlirPassManagerCreate, mlirPassManagerDestroy,
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
        dialect_registry::DialectRegistry, location::Location, utility::register_all_dialects,
    };
    use indoc::indoc;

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
    fn run() {
        let context = Context::new();
        let manager = PassManager::new(&context);

        manager.add_pass(Pass::convert_func_to_llvm());
        manager.run(&Module::new(Location::unknown(&context)));
    }

    fn register_all_upstream_dialects(context: &Context) {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
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
        );

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
        );

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

    // void testParsePassPipeline() {
    //   MlirContext ctx = mlirContextCreate();
    //   MlirPassManager pm = mlirPassManagerCreate(ctx);
    //   // Try parse a pipeline.
    //   MlirLogicalResult status = mlirParsePassPipeline(
    //       mlirPassManagerGetAsOpPassManager(pm),
    //       mlirStringRefCreateFromCString(
    //           "builtin.module(func.func(print-op-stats{json=false}),"
    //           " func.func(print-op-stats{json=false}))"));
    //   // Expect a failure, we haven't registered the print-op-stats pass yet.
    //   if (mlirLogicalResultIsSuccess(status)) {
    //     fprintf(
    //         stderr,
    //         "Unexpected success parsing pipeline without registering the pass\n");
    //     exit(EXIT_FAILURE);
    //   }
    //   // Try again after registrating the pass.
    //   mlirRegisterTransformsPrintOpStats();
    //   status = mlirParsePassPipeline(
    //       mlirPassManagerGetAsOpPassManager(pm),
    //       mlirStringRefCreateFromCString(
    //           "builtin.module(func.func(print-op-stats{json=false}),"
    //           " func.func(print-op-stats{json=false}))"));
    //   // Expect a failure, we haven't registered the print-op-stats pass yet.
    //   if (mlirLogicalResultIsFailure(status)) {
    //     fprintf(stderr,
    //             "Unexpected failure parsing pipeline after registering the pass\n");
    //     exit(EXIT_FAILURE);
    //   }

    //   // CHECK: Round-trip: builtin.module(func.func(print-op-stats{json=false}),
    //   // func.func(print-op-stats{json=false}))
    //   fprintf(stderr, "Round-trip: ");
    //   mlirPrintPassPipeline(mlirPassManagerGetAsOpPassManager(pm), printToStderr,
    //                         NULL);
    //   fprintf(stderr, "\n");
    //   mlirPassManagerDestroy(pm);
    //   mlirContextDestroy(ctx);
    // }

    // struct TestExternalPassUserData {
    //   int constructCallCount;
    //   int destructCallCount;
    //   int initializeCallCount;
    //   int cloneCallCount;
    //   int runCallCount;
    // };
    // typedef struct TestExternalPassUserData TestExternalPassUserData;

    // void testConstructExternalPass(void *userData) {
    //   ++((TestExternalPassUserData *)userData)->constructCallCount;
    // }

    // void testDestructExternalPass(void *userData) {
    //   ++((TestExternalPassUserData *)userData)->destructCallCount;
    // }

    // MlirLogicalResult testInitializeExternalPass(MlirContext ctx, void *userData) {
    //   ++((TestExternalPassUserData *)userData)->initializeCallCount;
    //   return mlirLogicalResultSuccess();
    // }

    // MlirLogicalResult testInitializeFailingExternalPass(MlirContext ctx,
    //                                                     void *userData) {
    //   ++((TestExternalPassUserData *)userData)->initializeCallCount;
    //   return mlirLogicalResultFailure();
    // }

    // void *testCloneExternalPass(void *userData) {
    //   ++((TestExternalPassUserData *)userData)->cloneCallCount;
    //   return userData;
    // }

    // void testRunExternalPass(MlirOperation op, MlirExternalPass pass,
    //                          void *userData) {
    //   ++((TestExternalPassUserData *)userData)->runCallCount;
    // }

    // void testRunExternalFuncPass(MlirOperation op, MlirExternalPass pass,
    //                              void *userData) {
    //   ++((TestExternalPassUserData *)userData)->runCallCount;
    //   MlirStringRef opName = mlirIdentifierStr(mlirOperationGetName(op));
    //   if (!mlirStringRefEqual(opName,
    //                           mlirStringRefCreateFromCString("func.func"))) {
    //     mlirExternalPassSignalFailure(pass);
    //   }
    // }

    // void testRunFailingExternalPass(MlirOperation op, MlirExternalPass pass,
    //                                 void *userData) {
    //   ++((TestExternalPassUserData *)userData)->runCallCount;
    //   mlirExternalPassSignalFailure(pass);
    // }

    // MlirExternalPassCallbacks makeTestExternalPassCallbacks(
    //     MlirLogicalResult (*initializePass)(MlirContext ctx, void *userData),
    //     void (*runPass)(MlirOperation op, MlirExternalPass, void *userData)) {
    //   return (MlirExternalPassCallbacks){testConstructExternalPass,
    //                                      testDestructExternalPass, initializePass,
    //                                      testCloneExternalPass, runPass};
    // }

    // void testExternalPass() {
    //   MlirContext ctx = mlirContextCreate();
    //   registerAllUpstreamDialects(ctx);

    //   MlirModule module = mlirModuleCreateParse(
    //       ctx,
    //       // clang-format off
    //       mlirStringRefCreateFromCString(
    // "func.func @foo(%arg0 : i32) -> i32 {                                   \n"
    // "  %res = arith.addi %arg0, %arg0 : i32                                     \n"
    // "  return %res : i32                                                        \n"
    // "}"));
    //   // clang-format on
    //   if (mlirModuleIsNull(module)) {
    //     fprintf(stderr, "Unexpected failure parsing module.\n");
    //     exit(EXIT_FAILURE);
    //   }

    //   MlirStringRef description = mlirStringRefCreateFromCString("");
    //   MlirStringRef emptyOpName = mlirStringRefCreateFromCString("");

    //   MlirTypeIDAllocator typeIDAllocator = mlirTypeIDAllocatorCreate();

    //   // Run a generic pass
    //   {
    //     MlirTypeID passID = mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    //     MlirStringRef name = mlirStringRefCreateFromCString("TestExternalPass");
    //     MlirStringRef argument =
    //         mlirStringRefCreateFromCString("test-external-pass");
    //     TestExternalPassUserData userData = {0};

    //     MlirPass externalPass = mlirCreateExternalPass(
    //         passID, name, argument, description, emptyOpName, 0, NULL,
    //         makeTestExternalPassCallbacks(NULL, testRunExternalPass), &userData);

    //     if (userData.constructCallCount != 1) {
    //       fprintf(stderr, "Expected constructCallCount to be 1\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     MlirPassManager pm = mlirPassManagerCreate(ctx);
    //     mlirPassManagerAddOwnedPass(pm, externalPass);
    //     MlirLogicalResult success = mlirPassManagerRun(pm, module);
    //     if (mlirLogicalResultIsFailure(success)) {
    //       fprintf(stderr, "Unexpected failure running external pass.\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     if (userData.runCallCount != 1) {
    //       fprintf(stderr, "Expected runCallCount to be 1\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     mlirPassManagerDestroy(pm);

    //     if (userData.destructCallCount != userData.constructCallCount) {
    //       fprintf(stderr, "Expected destructCallCount to be equal to "
    //                       "constructCallCount\n");
    //       exit(EXIT_FAILURE);
    //     }
    //   }

    //   // Run a func operation pass
    //   {
    //     MlirTypeID passID = mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    //     MlirStringRef name = mlirStringRefCreateFromCString("TestExternalFuncPass");
    //     MlirStringRef argument =
    //         mlirStringRefCreateFromCString("test-external-func-pass");
    //     TestExternalPassUserData userData = {0};
    //     MlirDialectHandle funcHandle = mlirGetDialectHandle__func__();
    //     MlirStringRef funcOpName = mlirStringRefCreateFromCString("func.func");

    //     MlirPass externalPass = mlirCreateExternalPass(
    //         passID, name, argument, description, funcOpName, 1, &funcHandle,
    //         makeTestExternalPassCallbacks(NULL, testRunExternalFuncPass),
    //         &userData);

    //     if (userData.constructCallCount != 1) {
    //       fprintf(stderr, "Expected constructCallCount to be 1\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     MlirPassManager pm = mlirPassManagerCreate(ctx);
    //     MlirOpPassManager nestedFuncPm =
    //         mlirPassManagerGetNestedUnder(pm, funcOpName);
    //     mlirOpPassManagerAddOwnedPass(nestedFuncPm, externalPass);
    //     MlirLogicalResult success = mlirPassManagerRun(pm, module);
    //     if (mlirLogicalResultIsFailure(success)) {
    //       fprintf(stderr, "Unexpected failure running external operation pass.\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     // Since this is a nested pass, it can be cloned and run in parallel
    //     if (userData.cloneCallCount != userData.constructCallCount - 1) {
    //       fprintf(stderr, "Expected constructCallCount to be 1\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     // The pass should only be run once this there is only one func op
    //     if (userData.runCallCount != 1) {
    //       fprintf(stderr, "Expected runCallCount to be 1\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     mlirPassManagerDestroy(pm);

    //     if (userData.destructCallCount != userData.constructCallCount) {
    //       fprintf(stderr, "Expected destructCallCount to be equal to "
    //                       "constructCallCount\n");
    //       exit(EXIT_FAILURE);
    //     }
    //   }

    //   // Run a pass with `initialize` set
    //   {
    //     MlirTypeID passID = mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    //     MlirStringRef name = mlirStringRefCreateFromCString("TestExternalPass");
    //     MlirStringRef argument =
    //         mlirStringRefCreateFromCString("test-external-pass");
    //     TestExternalPassUserData userData = {0};

    //     MlirPass externalPass = mlirCreateExternalPass(
    //         passID, name, argument, description, emptyOpName, 0, NULL,
    //         makeTestExternalPassCallbacks(testInitializeExternalPass,
    //                                       testRunExternalPass),
    //         &userData);

    //     if (userData.constructCallCount != 1) {
    //       fprintf(stderr, "Expected constructCallCount to be 1\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     MlirPassManager pm = mlirPassManagerCreate(ctx);
    //     mlirPassManagerAddOwnedPass(pm, externalPass);
    //     MlirLogicalResult success = mlirPassManagerRun(pm, module);
    //     if (mlirLogicalResultIsFailure(success)) {
    //       fprintf(stderr, "Unexpected failure running external pass.\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     if (userData.initializeCallCount != 1) {
    //       fprintf(stderr, "Expected initializeCallCount to be 1\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     if (userData.runCallCount != 1) {
    //       fprintf(stderr, "Expected runCallCount to be 1\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     mlirPassManagerDestroy(pm);

    //     if (userData.destructCallCount != userData.constructCallCount) {
    //       fprintf(stderr, "Expected destructCallCount to be equal to "
    //                       "constructCallCount\n");
    //       exit(EXIT_FAILURE);
    //     }
    //   }

    //   // Run a pass that fails during `initialize`
    //   {
    //     MlirTypeID passID = mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    //     MlirStringRef name =
    //         mlirStringRefCreateFromCString("TestExternalFailingPass");
    //     MlirStringRef argument =
    //         mlirStringRefCreateFromCString("test-external-failing-pass");
    //     TestExternalPassUserData userData = {0};

    //     MlirPass externalPass = mlirCreateExternalPass(
    //         passID, name, argument, description, emptyOpName, 0, NULL,
    //         makeTestExternalPassCallbacks(testInitializeFailingExternalPass,
    //                                       testRunExternalPass),
    //         &userData);

    //     if (userData.constructCallCount != 1) {
    //       fprintf(stderr, "Expected constructCallCount to be 1\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     MlirPassManager pm = mlirPassManagerCreate(ctx);
    //     mlirPassManagerAddOwnedPass(pm, externalPass);
    //     MlirLogicalResult success = mlirPassManagerRun(pm, module);
    //     if (mlirLogicalResultIsSuccess(success)) {
    //       fprintf(
    //           stderr,
    //           "Expected failure running pass manager on failing external pass.\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     if (userData.initializeCallCount != 1) {
    //       fprintf(stderr, "Expected initializeCallCount to be 1\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     if (userData.runCallCount != 0) {
    //       fprintf(stderr, "Expected runCallCount to be 0\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     mlirPassManagerDestroy(pm);

    //     if (userData.destructCallCount != userData.constructCallCount) {
    //       fprintf(stderr, "Expected destructCallCount to be equal to "
    //                       "constructCallCount\n");
    //       exit(EXIT_FAILURE);
    //     }
    //   }

    //   // Run a pass that fails during `run`
    //   {
    //     MlirTypeID passID = mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    //     MlirStringRef name =
    //         mlirStringRefCreateFromCString("TestExternalFailingPass");
    //     MlirStringRef argument =
    //         mlirStringRefCreateFromCString("test-external-failing-pass");
    //     TestExternalPassUserData userData = {0};

    //     MlirPass externalPass = mlirCreateExternalPass(
    //         passID, name, argument, description, emptyOpName, 0, NULL,
    //         makeTestExternalPassCallbacks(NULL, testRunFailingExternalPass),
    //         &userData);

    //     if (userData.constructCallCount != 1) {
    //       fprintf(stderr, "Expected constructCallCount to be 1\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     MlirPassManager pm = mlirPassManagerCreate(ctx);
    //     mlirPassManagerAddOwnedPass(pm, externalPass);
    //     MlirLogicalResult success = mlirPassManagerRun(pm, module);
    //     if (mlirLogicalResultIsSuccess(success)) {
    //       fprintf(
    //           stderr,
    //           "Expected failure running pass manager on failing external pass.\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     if (userData.runCallCount != 1) {
    //       fprintf(stderr, "Expected runCallCount to be 1\n");
    //       exit(EXIT_FAILURE);
    //     }

    //     mlirPassManagerDestroy(pm);

    //     if (userData.destructCallCount != userData.constructCallCount) {
    //       fprintf(stderr, "Expected destructCallCount to be equal to "
    //                       "constructCallCount\n");
    //       exit(EXIT_FAILURE);
    //     }
    //   }

    //   mlirTypeIDAllocatorDestroy(typeIDAllocator);
    //   mlirModuleDestroy(module);
    //   mlirContextDestroy(ctx);
    // }
}
