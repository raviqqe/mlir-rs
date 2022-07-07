use mlir_sys::MlirValue;

pub struct Value {
    value: MlirValue,
}

impl Value {
    pub(crate) unsafe fn from_raw(value: MlirValue) -> Self {
        Self { value }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirValue {
        self.value
    }
}
