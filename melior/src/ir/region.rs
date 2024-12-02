use super::{Block, BlockRef};
use mlir_sys::{
    mlirRegionAppendOwnedBlock, mlirRegionCreate, mlirRegionDestroy, mlirRegionEqual,
    mlirRegionGetFirstBlock, mlirRegionInsertOwnedBlockAfter, mlirRegionInsertOwnedBlockBefore,
    MlirRegion,
};
use std::{
    marker::PhantomData,
    mem::{forget, transmute},
    ops::Deref,
};

/// A region.
#[derive(Debug)]
pub struct Region<'c> {
    raw: MlirRegion,
    _block: PhantomData<Block<'c>>,
}

impl<'c> Region<'c> {
    /// Creates a region.
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirRegionCreate() },
            _block: Default::default(),
        }
    }

    /// Returns the first block in a region.
    pub fn first_block(&self) -> Option<BlockRef<'c, '_>> {
        unsafe {
            let block = mlirRegionGetFirstBlock(self.raw);

            if block.ptr.is_null() {
                None
            } else {
                Some(BlockRef::from_raw(block))
            }
        }
    }

    /// Inserts a block after another block.
    pub fn insert_block_after(&self, one: BlockRef<'c, '_>, other: Block<'c>) -> BlockRef<'c, '_> {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());

            mlirRegionInsertOwnedBlockAfter(self.raw, one.to_raw(), other.into_raw());

            r#ref
        }
    }

    /// Inserts a block before another block.
    pub fn insert_block_before(&self, one: BlockRef<'c, '_>, other: Block<'c>) -> BlockRef<'c, '_> {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());

            mlirRegionInsertOwnedBlockBefore(self.raw, one.to_raw(), other.into_raw());

            r#ref
        }
    }

    /// Appends a block.
    pub fn append_block(&self, block: Block<'c>) -> BlockRef<'c, '_> {
        unsafe {
            let r#ref = BlockRef::from_raw(block.to_raw());

            mlirRegionAppendOwnedBlock(self.raw, block.into_raw());

            r#ref
        }
    }

    /// Converts a region into a raw object.
    pub fn into_raw(self) -> mlir_sys::MlirRegion {
        let region = self.raw;

        forget(self);

        region
    }
}

impl Default for Region<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Region<'_> {
    fn drop(&mut self) {
        unsafe { mlirRegionDestroy(self.raw) }
    }
}

impl PartialEq for Region<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}

impl Eq for Region<'_> {}

/// A reference to a region.
#[derive(Clone, Copy, Debug)]
pub struct RegionRef<'c, 'a> {
    raw: MlirRegion,
    _region: PhantomData<&'a Region<'c>>,
}

impl RegionRef<'_, '_> {
    /// Creates a region from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirRegion) -> Self {
        Self {
            raw,
            _region: Default::default(),
        }
    }

    /// Creates an optional region from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_option_raw(raw: MlirRegion) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}

impl<'c> Deref for RegionRef<'c, '_> {
    type Target = Region<'c>;

    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}

impl PartialEq for RegionRef<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}

impl Eq for RegionRef<'_, '_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Region::new();
    }

    #[test]
    fn first_block() {
        assert!(Region::new().first_block().is_none());
    }

    #[test]
    fn append_block() {
        let region = Region::new();
        let block = Block::new(&[]);

        region.append_block(block);

        assert!(region.first_block().is_some());
    }

    #[test]
    fn insert_block_after() {
        let region = Region::new();

        let block = region.append_block(Block::new(&[]));
        region.insert_block_after(block, Block::new(&[]));

        assert_eq!(region.first_block(), Some(block));
    }

    #[test]
    fn insert_block_before() {
        let region = Region::new();

        let block = region.append_block(Block::new(&[]));
        let block = region.insert_block_before(block, Block::new(&[]));

        assert_eq!(region.first_block(), Some(block));
    }

    #[test]
    fn equal() {
        let region = Region::new();

        assert_eq!(region, region);
    }

    #[test]
    fn not_equal() {
        assert_ne!(Region::new(), Region::new());
    }
}
