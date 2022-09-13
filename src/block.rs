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

/// A block
pub struct Block<'c> {
    raw: MlirBlock,
    _context: PhantomData<&'c Context>,
}

impl<'c> Block<'c> {
    /// Creates a block.
    pub fn new(arguments: &[(Type<'c>, Location<'c>)]) -> Self {
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

    /// Gets an argument at a position.
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

    /// Gets a parent region.
    pub fn parent_region(&self) -> RegionRef {
        unsafe { RegionRef::from_raw(mlirBlockGetParentRegion(self.raw)) }
    }

    /// Gets the first operation.
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

    /// Adds an argument.
    pub fn add_argument(&self, r#type: Type<'c>, location: Location<'c>) -> Value {
        unsafe {
            Value::from_raw(mlirBlockAddArgument(
                self.raw,
                r#type.to_raw(),
                location.to_raw(),
            ))
        }
    }

    /// Inserts an operation.
    // TODO How can we make those update functions take `&mut self`?
    // TODO Use cells?
    pub fn insert_operation(&self, position: usize, operation: Operation) -> OperationRef {
        unsafe {
            let operation = operation.into_raw();

            mlirBlockInsertOwnedOperation(self.raw, position as isize, operation);

            OperationRef::from_raw(operation)
        }
    }

    /// Appends an operation.
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

    pub(crate) unsafe fn to_raw(&self) -> MlirBlock {
        self.raw
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Block::new(&[]);
    }

    #[test]
    fn get_non_existent_argument() {
        assert!(Block::new(&[]).argument(0).is_none());
    }
}
