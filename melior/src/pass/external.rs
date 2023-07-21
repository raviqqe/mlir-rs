//! External passes

use super::Pass;
use crate::{
    dialect::DialectHandle,
    ir::{r#type::TypeId, OperationRef},
    ContextRef, StringRef,
};
use mlir_sys::{
    mlirCreateExternalPass, MlirContext, MlirExternalPass, MlirExternalPassCallbacks,
    MlirLogicalResult, MlirOperation,
};

unsafe extern "C" fn callback_construct<'a, T: ExternalPass<'a>>(pass: *mut T) {
    pass.as_mut()
        .expect("pass should be valid when called")
        .construct();
}

unsafe extern "C" fn callback_destruct<'a, T: ExternalPass<'a>>(pass: *mut T) {
    pass.as_mut()
        .expect("pass should be valid when called")
        .destruct();
    std::ptr::drop_in_place(pass);
}

unsafe extern "C" fn callback_initialize<'a, T: ExternalPass<'a>>(
    ctx: MlirContext,
    pass: *mut T,
) -> MlirLogicalResult {
    pass.as_mut()
        .expect("pass should be valid when called")
        .initialize(ContextRef::from_raw(ctx));
    MlirLogicalResult { value: 1 }
}

unsafe extern "C" fn callback_run<'a, T: ExternalPass<'a>>(
    op: MlirOperation,
    _mlir_pass: MlirExternalPass,
    pass: *mut T,
) {
    pass.as_mut()
        .expect("pass should be valid when called")
        .run(OperationRef::from_raw(op))
}

unsafe extern "C" fn callback_clone<'a, T: ExternalPass<'a>>(pass: *mut T) -> *mut T {
    Box::<T>::into_raw(Box::new(
        pass.as_mut()
            .expect("pass should be valid when called")
            .clone(),
    ))
}

/// A trait for MLIR passes written in Rust.
///
/// This trait is implemented for any type that implements `FnMut`,
/// but can be implemented for any struct that implements `Clone`.
///
/// # Examples
///
/// The following example pass dumps operations.
///
/// ```
/// use melior::{
///     ir::OperationRef,
///     pass::ExternalPass,
///     ContextRef,
/// };
///
/// #[derive(Clone, Debug)]
/// struct ExamplePass;
///
/// impl<'c> ExternalPass<'c> for ExamplePass {
///     fn construct(&mut self) {
///         println!("Constructed pass!");
///     }
///
///     fn initialize(&mut self, context: ContextRef<'c>) {
///         println!("Initialize called!");
///     }
///
///     fn run(&mut self, operation: OperationRef<'c, '_>) {
///         operation.dump();
///     }
/// }
/// ```
pub trait ExternalPass<'c>: Sized + Clone {
    fn construct(&mut self) {}
    fn destruct(&mut self) {}
    fn initialize(&mut self, context: ContextRef<'c>);
    fn run(&mut self, operation: OperationRef<'c, '_>);
}

impl<'c, F: FnMut(OperationRef<'c, '_>) + Clone> ExternalPass<'c> for F {
    fn initialize(&mut self, _context: ContextRef<'c>) {}

    fn run(&mut self, operation: OperationRef<'c, '_>) {
        self(operation)
    }
}

/// Creates a `Pass` object from an external pass
///
/// # Examples
///
/// ```
/// use melior::{
///     ir::{r#type::TypeId, OperationRef},
///     pass::create_external,
/// };
///
/// #[repr(align(8))]
/// struct PassId;
///
/// static EXAMPLE_PASS: PassId = PassId;
///
/// create_external(
///     |operation: OperationRef| {
///         operation.dump();
///     },
///     TypeId::create(&EXAMPLE_PASS),
///     "name",
///     "argument",
///     "description",
///     "",
///     &[],
/// );
/// ```
pub fn create_external<'c, T: ExternalPass<'c>>(
    pass: T,
    pass_id: TypeId,
    name: &str,
    argument: &str,
    description: &str,
    op_name: &str,
    dependent_dialects: &[DialectHandle],
) -> Pass {
    unsafe {
        let mut dep_dialects_raw: Vec<_> = dependent_dialects.iter().map(|d| d.to_raw()).collect();
        let callbacks = MlirExternalPassCallbacks {
            construct: Some(std::mem::transmute(callback_construct::<T> as *const ())),
            destruct: Some(std::mem::transmute(callback_destruct::<T> as *const ())),
            initialize: Some(std::mem::transmute(callback_initialize::<T> as *const ())),
            run: Some(std::mem::transmute(callback_run::<T> as *const ())),
            clone: Some(std::mem::transmute(callback_clone::<T> as *const ())),
        };
        let pass_box = Box::<T>::into_raw(Box::new(pass));
        let raw_pass = mlirCreateExternalPass(
            pass_id.to_raw(),
            StringRef::from(name).to_raw(),
            StringRef::from(argument).to_raw(),
            StringRef::from(description).to_raw(),
            StringRef::from(op_name).to_raw(),
            dep_dialects_raw.len() as isize,
            dep_dialects_raw.as_mut_ptr(),
            callbacks,
            pass_box as *mut std::ffi::c_void,
        );
        Pass::from_raw(raw_pass)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        dialect::func,
        ir::{
            attribute::{StringAttribute, TypeAttribute},
            r#type::FunctionType,
            Block, Identifier, Location, Module, Region,
        },
        pass::PassManager,
        test::create_test_context,
        Context,
    };

    use super::*;

    #[repr(align(8))]
    struct PassId;

    fn create_module(context: &Context) -> Module {
        let location = Location::unknown(&context);
        let module = Module::new(location);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));
        module
    }

    #[test]
    fn external_pass() {
        static TEST_PASS: PassId = PassId;

        #[derive(Clone, Debug)]
        struct TestPass {
            value: i32,
        }

        impl<'c> ExternalPass<'c> for TestPass {
            fn construct(&mut self) {
                assert_eq!(self.value, 10);
            }

            fn destruct(&mut self) {
                assert_eq!(self.value, 30);
            }

            fn initialize(&mut self, _context: ContextRef<'c>) {
                assert_eq!(self.value, 10);
                self.value = 20;
            }

            fn run(&mut self, operation: OperationRef<'c, '_>) {
                assert_eq!(self.value, 20);
                self.value = 30;
                assert!(operation.verify());
                assert!(
                    operation
                        .region(0)
                        .expect("module has a body")
                        .first_block()
                        .expect("module has a body")
                        .first_operation()
                        .expect("body has a function")
                        .name()
                        == Identifier::new(&operation.context(), "func.func")
                );
            }
        }

        impl TestPass {
            fn create(self) -> Pass {
                create_external(
                    self,
                    TypeId::create(&TEST_PASS),
                    "test pass",
                    "test argument",
                    "a test pass",
                    "",
                    &[DialectHandle::func()],
                )
            }
        }

        let context = create_test_context();

        let mut module = create_module(&context);
        let pass_manager = PassManager::new(&context);

        let test_pass = TestPass { value: 10 };
        pass_manager.add_pass(test_pass.create());
        pass_manager.run(&mut module).unwrap();
    }

    #[test]
    fn external_fn_pass() {
        static TEST_FN_PASS: PassId = PassId;

        let context = create_test_context();

        let mut module = create_module(&context);
        let pass_manager = PassManager::new(&context);

        pass_manager.add_pass(create_external(
            |operation: OperationRef| {
                assert!(operation.verify());
                assert!(
                    operation
                        .region(0)
                        .expect("module has a body")
                        .first_block()
                        .expect("module has a body")
                        .first_operation()
                        .expect("body has a function")
                        .name()
                        == Identifier::new(&operation.context(), "func.func")
                );
            },
            TypeId::create(&TEST_FN_PASS),
            "test closure",
            "test argument",
            "test",
            "",
            &[DialectHandle::func()],
        ));
        pass_manager.run(&mut module).unwrap();
    }
}
