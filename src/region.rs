use crate::block::{Block, BlockRef};
use mlir_sys::{
    mlirRegionAppendOwnedBlock, mlirRegionCreate, mlirRegionDestroy, mlirRegionGetFirstBlock,
    MlirRegion,
};
use std::{
    marker::PhantomData,
    mem::{forget, ManuallyDrop},
    ops::Deref,
};

pub struct Region {
    region: MlirRegion,
}

impl Region {
    pub fn new() -> Self {
        Self {
            region: unsafe { mlirRegionCreate() },
        }
    }

    pub fn first_block(&self) -> BlockRef {
        unsafe { BlockRef::from_raw(mlirRegionGetFirstBlock(self.region)) }
    }

    pub fn append_block(&self, block: Block) {
        unsafe { mlirRegionAppendOwnedBlock(self.region, block.into_raw()) }
    }

    pub(crate) unsafe fn into_raw(self) -> mlir_sys::MlirRegion {
        let region = self.region;

        forget(self);

        region
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        unsafe { mlirRegionDestroy(self.region) }
    }
}

pub struct RegionRef<'a> {
    region: ManuallyDrop<Region>,
    _region: PhantomData<&'a Region>,
}

impl<'a> RegionRef<'a> {
    pub(crate) unsafe fn from_raw(region: MlirRegion) -> Self {
        Self {
            region: ManuallyDrop::new(Region { region }),
            _region: Default::default(),
        }
    }
}

impl<'a> Deref for RegionRef<'a> {
    type Target = Region;

    fn deref(&self) -> &Self::Target {
        &self.region
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
