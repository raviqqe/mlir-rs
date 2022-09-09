use mlir_sys::{
    mlirDialectRegistryCreate, mlirDialectRegistryDestroy, mlirRegisterAllDialects,
    MlirDialectRegistry,
};

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
