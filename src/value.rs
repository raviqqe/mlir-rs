use mlir_sys::MlirValue;
use std::marker::PhantomData;

// Values are always non-owning references.
pub struct Value<'a> {
    value: MlirValue,
    _parent: PhantomData<&'a ()>,
}

impl<'a> Value<'a> {
    pub(crate) fn from_raw(value: MlirValue) -> Self {
        Self {
            value,
            _parent: Default::default(),
        }
    }

    pub(crate) fn to_raw(&self) -> MlirValue {
        self.value
    }
}
