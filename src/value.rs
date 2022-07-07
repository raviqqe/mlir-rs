use mlir_sys::MlirValue;

pub struct Value {
    value: MlirValue,
}

impl Value {
    pub(crate) fn from_raw(value: MlirValue) -> Self {
        Self { value }
    }

    pub(crate) fn to_raw(&self) -> MlirValue {
        self.value
    }
}
