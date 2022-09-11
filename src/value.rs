use mlir_sys::MlirValue;
use std::marker::PhantomData;

// Values are always non-owning references to their parents, such as operations and block arguments.
// See the `Value` class in the MLIR C++ API.
#[derive(Clone, Copy, Debug)]
pub struct Value<'a> {
    value: MlirValue,
    _parent: PhantomData<&'a ()>,
}

impl<'a> Value<'a> {
    pub(crate) unsafe fn from_raw(value: MlirValue) -> Self {
        Self {
            value,
            _parent: Default::default(),
        }
    }

    pub(crate) unsafe fn to_raw(self) -> MlirValue {
        self.value
    }
}
