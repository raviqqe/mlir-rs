use mlir_sys::{mlirDialectRegistryCreate, mlirDialectRegistryDestroy, MlirDialectRegistry};

/// A dialect registry.
#[derive(Debug)]
pub struct Registry {
    raw: MlirDialectRegistry,
}

impl Registry {
    /// Creates a dialect registry.
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirDialectRegistryCreate() },
        }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirDialectRegistry {
        self.raw
    }
}

impl Drop for Registry {
    fn drop(&mut self) {
        unsafe { mlirDialectRegistryDestroy(self.raw) };
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;

    #[test]
    fn new() {
        Registry::new();
    }

    #[test]
    fn register_all_dialects() {
        Registry::new();
    }

    #[test]
    fn register_dialect() {
        let registry = Registry::new();
        Handle::func().insert_dialect(&registry);

        let context = Context::new();
        let count = context.registered_dialect_count();

        context.append_dialect_registry(&registry);

        assert_eq!(context.registered_dialect_count() - count, 1);
    }
}
