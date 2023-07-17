use super::Pass;
use crate::{
    dialect::DialectHandle,
    ir::{r#type::TypeId, OperationRef},
    ContextRef, StringRef,
};
use mlir_sys::{MlirContext, MlirExternalPass, MlirLogicalResult, MlirOperation};

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
        let callbacks = mlir_sys::MlirExternalPassCallbacks {
            construct: Some(std::mem::transmute(callback_construct::<T> as *const ())),
            destruct: Some(std::mem::transmute(callback_destruct::<T> as *const ())),
            initialize: Some(std::mem::transmute(callback_initialize::<T> as *const ())),
            run: Some(std::mem::transmute(callback_run::<T> as *const ())),
            clone: Some(std::mem::transmute(callback_clone::<T> as *const ())),
        };
        let pass_box = Box::<T>::into_raw(Box::new(pass));
        let raw_pass = mlir_sys::mlirCreateExternalPass(
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
