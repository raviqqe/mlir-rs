use crate::{
    context::Context, location::Location, r#type::Type, region::RegionRef, utility::into_raw_array,
};
use mlir_sys::{mlirBlockCreate, mlirBlockDestroy, mlirBlockGetParentRegion, MlirBlock};
use std::marker::PhantomData;

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

    pub(crate) fn to_raw(&self) -> MlirBlock {
        self.block
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
