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

/// A region.
pub struct Region {
    raw: MlirRegion,
}

impl Region {
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirRegionCreate() },
        }
    }

    pub fn first_block(&self) -> Option<BlockRef> {
        unsafe {
            let block = mlirRegionGetFirstBlock(self.raw);

            if block.ptr.is_null() {
                None
            } else {
                Some(BlockRef::from_raw(block))
            }
        }
    }

    pub fn append_block(&self, block: Block) {
        unsafe { mlirRegionAppendOwnedBlock(self.raw, block.into_raw()) }
    }

    pub(crate) unsafe fn into_raw(self) -> mlir_sys::MlirRegion {
        let region = self.raw;

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
        unsafe { mlirRegionDestroy(self.raw) }
    }
}

pub struct RegionRef<'a> {
    raw: ManuallyDrop<Region>,
    _region: PhantomData<&'a Region>,
}

impl<'a> RegionRef<'a> {
    pub(crate) unsafe fn from_raw(region: MlirRegion) -> Self {
        Self {
            raw: ManuallyDrop::new(Region { raw: region }),
            _region: Default::default(),
        }
    }
}

impl<'a> Deref for RegionRef<'a> {
    type Target = Region;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Region::new();
    }

    #[test]
    fn first_block_none() {
        assert!(Region::new().first_block().is_none());
    }
}
