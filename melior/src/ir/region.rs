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

    /// Gets the first block in a region.
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

    /// Inserts a block after another block.
    pub fn insert_block_after(&self, one: BlockRef<'c, '_>, other: Block<'c>) -> BlockRef {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());

            mlirRegionInsertOwnedBlockAfter(self.raw, one.to_raw(), other.into_raw());

            r#ref
        }
    }

    /// Inserts a block before another block.
    pub fn insert_block_before(&self, one: BlockRef, other: Block) -> BlockRef {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());

            mlirRegionInsertOwnedBlockBefore(self.raw, one.to_raw(), other.into_raw());

            r#ref
        }
    }

    /// Appends a block.
    pub fn append_block(&self, block: Block) -> BlockRef {
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

impl<'c> Default for Region<'c> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'c> Drop for Region<'c> {
    fn drop(&mut self) {
        unsafe { mlirRegionDestroy(self.raw) }
    }
}

impl<'c> PartialEq for Region<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}

impl<'c> Eq for Region<'c> {}

/// A reference to a region.
#[derive(Clone, Copy, Debug)]
pub struct RegionRef<'c, 'a> {
    raw: MlirRegion,
    _region: PhantomData<&'a Region<'c>>,
}

impl<'c, 'a> RegionRef<'c, 'a> {
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

impl<'c, 'a> Deref for RegionRef<'c, 'a> {
    type Target = Region<'c>;

    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}

impl<'c, 'a> PartialEq for RegionRef<'c, 'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}

impl<'c, 'a> Eq for RegionRef<'c, 'a> {}

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
