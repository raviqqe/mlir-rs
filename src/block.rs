use crate::{
    context::Context, location::Location, operation::Operation, r#type::Type, region::RegionRef,
    utility::into_raw_array,
};
use mlir_sys::{
    mlirBlockAppendOwnedOperation, mlirBlockCreate, mlirBlockDestroy, mlirBlockGetParentRegion,
    mlirBlockInsertOwnedOperation, MlirBlock,
};
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

    pub fn insert_operation(&self, position: usize, operation: Operation) {
        unsafe {
            mlirBlockInsertOwnedOperation(self.block, position as isize, operation.into_raw())
        }
    }

    pub fn append_operation(&self, operation: Operation) {
        unsafe { mlirBlockAppendOwnedOperation(self.block, operation.into_raw()) }
    }

    pub(crate) unsafe fn from_raw(block: MlirBlock) -> Self {
        Self {
            block,
            _context: Default::default(),
        }
    }

    pub(crate) unsafe fn into_raw(self) -> MlirBlock {
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
    _reference: PhantomData<&'a Block<'a>>,
}

impl<'a> BlockRef<'a> {
    pub(crate) unsafe fn from_raw(block: MlirBlock) -> Self {
        Self {
            block: ManuallyDrop::new(Block::from_raw(block)),
            _reference: Default::default(),
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
