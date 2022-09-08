use mlir_sys::{
    mlirDialectRegistryCreate, mlirDialectRegistryDestroy, mlirRegisterAllDialects,
    MlirDialectRegistry,
};
use std::{marker::PhantomData, mem::ManuallyDrop, ops::Deref};

pub struct DialectRegistry {
    registry: MlirDialectRegistry,
}

impl DialectRegistry {
    pub fn new() -> Self {
        Self {
            registry: unsafe { mlirDialectRegistryCreate() },
        }
    }

    pub fn register_all_dialects(&self) {
        unsafe { mlirRegisterAllDialects(self.to_raw()) }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirDialectRegistry {
        self.registry
    }
}

impl Drop for DialectRegistry {
    fn drop(&mut self) {
        unsafe { mlirDialectRegistryDestroy(self.registry) };
    }
}

impl Default for DialectRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub struct DialectRegistryRef<'r> {
    registry: ManuallyDrop<DialectRegistry>,
    _registry: PhantomData<&'r DialectRegistry>,
}

impl<'r> DialectRegistryRef<'r> {
    pub(crate) unsafe fn from_raw(registry: MlirDialectRegistry) -> Self {
        Self {
            registry: ManuallyDrop::new(DialectRegistry { registry }),
            _registry: Default::default(),
        }
    }
}

impl<'c> Deref for DialectRegistryRef<'c> {
    type Target = DialectRegistry;

    fn deref(&self) -> &Self::Target {
        &self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        DialectRegistry::new();
    }

    #[test]
    fn register_all_dialects() {
        DialectRegistry::new();
    }
}
