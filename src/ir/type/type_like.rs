use super::Id;
use crate::context::ContextRef;
use mlir_sys::{
    mlirTypeDump, mlirTypeGetContext, mlirTypeGetTypeID, mlirTypeIsAFunction, MlirType,
};

pub trait TypeLike<'c> {
    /// Converts a type into a raw type.
    fn to_raw(&self) -> MlirType;

    /// Gets a context.
    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirTypeGetContext(self.to_raw())) }
    }

    /// Gets an ID.
    fn id(&self) -> Id {
        unsafe { Id::from_raw(mlirTypeGetTypeID(self.to_raw())) }
    }

    /// Returns `true` if a type is a function.
    fn is_function(&self) -> bool {
        unsafe { mlirTypeIsAFunction(self.to_raw()) }
    }

    /// Dumps a type.
    fn dump(&self) {
        unsafe { mlirTypeDump(self.to_raw()) }
    }
}
