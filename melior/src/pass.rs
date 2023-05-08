//! Passes and pass managers.

pub mod conversion;
mod manager;
mod operation_manager;
pub mod transform;

pub use self::{manager::Manager, operation_manager::OperationManager};
use mlir_sys::MlirPass;

/// A pass.
pub struct Pass {
    raw: MlirPass,
}

impl Pass {
    pub(crate) fn from_raw_fn(create_raw: unsafe extern "C" fn() -> MlirPass) -> Self {
        Self {
            raw: unsafe { create_raw() },
        }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirPass {
        self.raw
    }

    pub fn _macro_from_raw_fn(create_raw: unsafe extern "C" fn() -> MlirPass) -> Self {
        Self::from_raw_fn(create_raw)
    }
}
