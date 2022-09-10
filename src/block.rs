use crate::{
    context::Context, location::Location, r#type::Type, region::RegionRef, utility::into_raw_array,
};
use mlir_sys::{mlirBlockCreate, mlirBlockDestroy, mlirBlockGetParentRegion, MlirBlock};
use std::{
    marker::PhantomData,
    mem::{forget, ManuallyDrop},
    ops::Deref,
};

pub struct Block<'c> {
    block: MlirBlock,
    _context: PhantomData<&'c Context>,
}

impl<'c> Block<'c> {
    pub fn new(arguments: Vec<(Type, Location)>) -> Self {
        unsafe {
            Self::from_raw(mlirBlockCreate(
                arguments.len() as isize,
                into_raw_array(
                    arguments
                        .iter()
                        .map(|(argument, _)| argument.to_raw())
                        .collect(),
                ),
                into_raw_array(
                    arguments
                        .iter()
                        .map(|(_, location)| location.to_raw())
                        .collect(),
                ),
            ))
        }
    }

    pub fn parent_region(&self) -> RegionRef {
        unsafe { RegionRef::from_raw(mlirBlockGetParentRegion(self.block)) }
    }

    pub(crate) fn from_raw(block: MlirBlock) -> Self {
        Self {
            block,
            _context: Default::default(),
        }
    }

    pub(crate) fn into_raw(self) -> MlirBlock {
        let block = self.block;

        forget(self);

        block
    }
}

impl<'c> Drop for Block<'c> {
    fn drop(&mut self) {
        unsafe { mlirBlockDestroy(self.block) };
    }
}

// TODO Should we split context lifetimes? Or, is it transitively proven that 'c > 'a?
pub struct BlockRef<'a> {
    block: ManuallyDrop<Block<'a>>,
    _block: PhantomData<&'a Block<'a>>,
}

impl<'a> BlockRef<'a> {
    pub(crate) unsafe fn from_raw(block: MlirBlock) -> Self {
        Self {
            block: ManuallyDrop::new(Block::from_raw(block)),
            _block: Default::default(),
        }
    }
}

impl<'a> Deref for BlockRef<'a> {
    type Target = Block<'a>;

    fn deref(&self) -> &Self::Target {
        &self.block
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Block::new(vec![]);
    }
}
