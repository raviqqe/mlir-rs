use mlir_sys::{mlirRegionCreate, MlirRegion};

pub struct Region {
    region: MlirRegion,
}

impl Region {
    pub fn new() -> Self {
        Self {
            region: unsafe { mlirRegionCreate() },
        }
    }

    pub(crate) fn from_raw(region: MlirRegion) -> Self {
        Self { region }
    }

    pub(crate) unsafe fn to_raw(&self) -> mlir_sys::MlirRegion {
        self.region
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Region::new();
    }
}
