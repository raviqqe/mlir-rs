pub mod conversion;
mod manager;
mod operation_manager;
pub mod transform;

pub use self::{manager::PassManager, operation_manager::OperationPassManager};
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
}
