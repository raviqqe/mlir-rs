use crate::{
    context::Context,
    location::Location,
    operation::{Operation, OperationRef},
    r#type::Type,
    region::RegionRef,
    utility::into_raw_array,
    value::Value,
};
use mlir_sys::{
    mlirBlockAddArgument, mlirBlockAppendOwnedOperation, mlirBlockCreate, mlirBlockDestroy,
    mlirBlockGetArgument, mlirBlockGetFirstOperation, mlirBlockGetNumArguments,
    mlirBlockGetParentRegion, mlirBlockInsertOwnedOperation, MlirBlock,
};
use std::{
    marker::PhantomData,
    mem::{forget, ManuallyDrop},
    ops::{Deref, DerefMut},
};

pub struct Block<'c> {
    raw: MlirBlock,
    _context: PhantomData<&'c Context>,
}

impl<'c> Block<'c> {
    pub fn new(arguments: Vec<(Type<'c>, Location)>) -> Self {
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

    pub fn argument(&self, position: usize) -> Option<Value> {
        unsafe {
            if position < mlirBlockGetNumArguments(self.raw) as usize {
                Some(Value::from_raw(mlirBlockGetArgument(
                    self.raw,
                    position as isize,
                )))
            } else {
                None
            }
        }
    }

    pub fn parent_region(&self) -> RegionRef {
        unsafe { RegionRef::from_raw(mlirBlockGetParentRegion(self.raw)) }
    }

    pub fn first_operation(&self) -> Option<OperationRef> {
        unsafe {
            let operation = mlirBlockGetFirstOperation(self.raw);

            if operation.ptr.is_null() {
                None
            } else {
                Some(OperationRef::from_raw(operation))
            }
        }
    }

    pub fn add_argument(&self, r#type: Type<'c>, location: Location<'c>) -> Value {
        unsafe {
            Value::from_raw(mlirBlockAddArgument(
                self.raw,
                r#type.to_raw(),
                location.to_raw(),
            ))
        }
    }

    // TODO How can we make those update functions take `&mut self`?
    // TODO Use cells?
    pub fn insert_operation(&self, position: usize, operation: Operation) -> OperationRef {
        unsafe {
            let operation = operation.into_raw();

            mlirBlockInsertOwnedOperation(self.raw, position as isize, operation);

            OperationRef::from_raw(operation)
        }
    }

    pub fn append_operation(&self, operation: Operation) -> OperationRef {
        unsafe {
            let operation = operation.into_raw();

            mlirBlockAppendOwnedOperation(self.raw, operation);

            OperationRef::from_raw(operation)
        }
    }

    pub(crate) unsafe fn from_raw(block: MlirBlock) -> Self {
        Self {
            raw: block,
            _context: Default::default(),
        }
    }

    pub(crate) unsafe fn into_raw(self) -> MlirBlock {
        let block = self.raw;

        forget(self);

        block
    }
}

impl<'c> Drop for Block<'c> {
    fn drop(&mut self) {
        unsafe { mlirBlockDestroy(self.raw) };
    }
}

// TODO Should we split context lifetimes? Or, is it transitively proven that 'c
// > 'a?
pub struct BlockRef<'a> {
    raw: ManuallyDrop<Block<'a>>,
    _reference: PhantomData<&'a Block<'a>>,
}

impl<'a> BlockRef<'a> {
    pub(crate) unsafe fn from_raw(block: MlirBlock) -> Self {
        Self {
            raw: ManuallyDrop::new(Block::from_raw(block)),
            _reference: Default::default(),
        }
    }
}

impl<'a> Deref for BlockRef<'a> {
    type Target = Block<'a>;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

pub struct BlockRefMut<'a> {
    raw: ManuallyDrop<Block<'a>>,
    _reference: PhantomData<&'a mut Block<'a>>,
}

impl<'a> BlockRefMut<'a> {
    pub(crate) unsafe fn from_raw(block: MlirBlock) -> Self {
        Self {
            raw: ManuallyDrop::new(Block::from_raw(block)),
            _reference: Default::default(),
        }
    }
}

impl<'a> Deref for BlockRefMut<'a> {
    type Target = Block<'a>;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

impl<'a> DerefMut for BlockRefMut<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Block::new(vec![]);
    }

    #[test]
    fn get_non_existent_argument() {
        assert!(Block::new(vec![]).argument(0).is_none());
    }
}
