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
    return MlirLogicalResult { value: 1 };
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

pub trait ExternalPass<'a>: Sized + Clone {
    fn create_external(self) -> Pass {
        unsafe {
            let dep_dialects = self.dependent_dialects();
            let mut dep_dialects_raw: Vec<_> =
                dep_dialects.into_iter().map(|d| d.to_raw()).collect();
            let callbacks = mlir_sys::MlirExternalPassCallbacks {
                construct: Some(std::mem::transmute(callback_construct::<Self> as *const ())),
                destruct: Some(std::mem::transmute(callback_destruct::<Self> as *const ())),
                initialize: Some(std::mem::transmute(
                    callback_initialize::<Self> as *const (),
                )),
                run: Some(std::mem::transmute(callback_run::<Self> as *const ())),
                clone: Some(std::mem::transmute(callback_clone::<Self> as *const ())),
            };
            let pass = Box::<Self>::into_raw(Box::new(self));
            let pass_ref = pass.as_ref().expect("pass is still valid");
            let raw_pass = mlir_sys::mlirCreateExternalPass(
                pass_ref.type_id().to_raw(),
                pass_ref.name().to_raw(),
                pass_ref.argument().to_raw(),
                pass_ref.description().to_raw(),
                pass_ref.op_name().to_raw(),
                dep_dialects_raw.len() as isize,
                dep_dialects_raw.as_mut_ptr(),
                callbacks,
                pass as *mut std::ffi::c_void,
            );
            Pass::from_raw(raw_pass)
        }
    }

    fn type_id(&self) -> TypeId;
    fn name(&self) -> StringRef<'a>;
    fn argument(&self) -> StringRef<'a>;
    fn description(&self) -> StringRef<'a>;
    fn op_name(&self) -> StringRef<'a>;
    fn dependent_dialects(&self) -> Vec<DialectHandle>;

    fn construct(&mut self) {}
    fn destruct(&mut self) {}
    fn initialize(&mut self, context: ContextRef<'a>);
    fn run(&mut self, operation: OperationRef);
}
