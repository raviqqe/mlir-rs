//! Blocks.

mod argument;

pub use self::argument::BlockArgument;
use super::{
    operation::OperationRefMut, Location, Operation, OperationRef, RegionRef, Type, TypeLike, Value,
};
use crate::{context::Context, utility::print_callback, Error};
use mlir_sys::{
    mlirBlockAddArgument, mlirBlockAppendOwnedOperation, mlirBlockCreate, mlirBlockDestroy,
    mlirBlockDetach, mlirBlockEqual, mlirBlockGetArgument, mlirBlockGetFirstOperation,
    mlirBlockGetNextInRegion, mlirBlockGetNumArguments, mlirBlockGetParentOperation,
    mlirBlockGetParentRegion, mlirBlockGetTerminator, mlirBlockInsertOwnedOperation,
    mlirBlockInsertOwnedOperationAfter, mlirBlockInsertOwnedOperationBefore, mlirBlockPrint,
    MlirBlock,
};
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
    mem::{forget, transmute},
    ops::Deref,
};

pub trait BlockApi<'c, 'v> {
    /// Returns an argument at a position.
    fn argument(&self, index: usize) -> Result<BlockArgument<'c, 'v>, Error>;
    /// Returns a number of arguments.
    fn argument_count(&self) -> usize;

    /// Returns a reference to the first operation.
    fn first_operation(&self) -> Option<OperationRef<'c, 'v>>;
    /// Returns a mutable reference to the first operation.
    fn first_operation_mut(&mut self) -> Option<OperationRefMut<'c, 'v>>;

    /// Returns a reference to a terminator operation.
    fn terminator(&self) -> Option<OperationRef<'c, 'v>>;
    /// Returns a mutable reference to a terminator operation.
    fn terminator_mut(&mut self) -> Option<OperationRefMut<'c, 'v>>;

    /// Returns a parent region.
    // TODO Store lifetime of regions in blocks, or create another type like
    // `InsertedBlockRef`?
    fn parent_region(&self) -> Option<RegionRef<'c, 'v>>;
    /// Returns a parent operation.
    fn parent_operation(&self) -> Option<OperationRef<'c, 'v>>;

    /// Adds an argument.
    fn add_argument(&self, r#type: Type<'c>, location: Location<'c>) -> Value<'c, 'v>;

    /// Appends an operation.
    fn append_operation(&self, operation: Operation<'c>) -> OperationRef<'c, 'v>;
    /// Inserts an operation.
    // TODO How can we make those update functions take `&mut self`?
    // TODO Use cells?
    fn insert_operation(&self, position: usize, operation: Operation<'c>) -> OperationRef<'c, 'v>;
    /// Inserts an operation after another.
    fn insert_operation_after(
        &self,
        one: OperationRef<'c, 'v>,
        other: Operation<'c>,
    ) -> OperationRef<'c, 'v>;
    /// Inserts an operation before another.
    fn insert_operation_before(
        &self,
        one: OperationRef<'c, 'v>,
        other: Operation<'c>,
    ) -> OperationRef<'c, 'v>;

    /// Returns a next block in a region.
    fn next_in_region(&self) -> Option<BlockRef<'c, 'v>>;
}

/// A block.
pub struct Block<'c> {
    raw: MlirBlock,
    _context: PhantomData<&'c Context>,
}

impl<'c> Block<'c> {
    /// Creates a block.
    // TODO Should we accept types and locations separately?
    pub fn new(arguments: &[(Type<'c>, Location<'c>)]) -> Self {
        unsafe {
            Self::from_raw(mlirBlockCreate(
                arguments.len() as isize,
                arguments
                    .iter()
                    .map(|(argument, _)| argument.to_raw())
                    .collect::<Vec<_>>()
                    .as_ptr() as *const _,
                arguments
                    .iter()
                    .map(|(_, location)| location.to_raw())
                    .collect::<Vec<_>>()
                    .as_ptr() as *const _,
            ))
        }
    }

    /// Detaches a block from a region and assumes its ownership.
    ///
    /// # Safety
    ///
    /// This function might invalidate existing references to the block if you
    /// drop it too early.
    // TODO Implement this for BlockRefMut instead and mark it safe.
    pub unsafe fn detach(&self) -> Option<Block<'c>> {
        if self.parent_region().is_some() {
            mlirBlockDetach(self.raw);

            Some(Block::from_raw(self.raw))
        } else {
            None
        }
    }

    /// Creates a block from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirBlock) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Converts a block into a raw object.
    pub const fn into_raw(self) -> MlirBlock {
        let block = self.raw;

        forget(self);

        block
    }

    /// Converts a block into a raw object.
    pub const fn to_raw(&self) -> MlirBlock {
        self.raw
    }
}

impl<'c, 'v> BlockApi<'c, 'v> for &'v Block<'c> {
    fn argument(&self, index: usize) -> Result<BlockArgument<'c, 'v>, Error> {
        unsafe {
            if index < self.argument_count() {
                Ok(BlockArgument::from_raw(mlirBlockGetArgument(
                    self.raw,
                    index as isize,
                )))
            } else {
                Err(Error::PositionOutOfBounds {
                    name: "block argument",
                    value: self.to_string(),
                    index,
                })
            }
        }
    }

    fn argument_count(&self) -> usize {
        unsafe { mlirBlockGetNumArguments(self.raw) as usize }
    }

    fn first_operation(&self) -> Option<OperationRef<'c, 'v>> {
        unsafe { OperationRef::from_option_raw(mlirBlockGetFirstOperation(self.raw)) }
    }

    fn first_operation_mut(&mut self) -> Option<OperationRefMut<'c, 'v>> {
        unsafe { OperationRefMut::from_option_raw(mlirBlockGetFirstOperation(self.raw)) }
    }

    fn terminator(&self) -> Option<OperationRef<'c, 'v>> {
        unsafe { OperationRef::from_option_raw(mlirBlockGetTerminator(self.raw)) }
    }

    fn terminator_mut(&mut self) -> Option<OperationRefMut<'c, 'v>> {
        unsafe { OperationRefMut::from_option_raw(mlirBlockGetTerminator(self.raw)) }
    }

    fn parent_region(&self) -> Option<RegionRef<'c, 'v>> {
        unsafe { RegionRef::from_option_raw(mlirBlockGetParentRegion(self.raw)) }
    }

    fn parent_operation(&self) -> Option<OperationRef<'c, 'v>> {
        unsafe { OperationRef::from_option_raw(mlirBlockGetParentOperation(self.raw)) }
    }

    fn add_argument(&self, r#type: Type<'c>, location: Location<'c>) -> Value<'c, 'v> {
        unsafe {
            Value::from_raw(mlirBlockAddArgument(
                self.raw,
                r#type.to_raw(),
                location.to_raw(),
            ))
        }
    }

    fn append_operation(&self, operation: Operation<'c>) -> OperationRef<'c, 'v> {
        unsafe {
            let operation = operation.into_raw();

            mlirBlockAppendOwnedOperation(self.raw, operation);

            OperationRef::from_raw(operation)
        }
    }

    fn insert_operation(&self, position: usize, operation: Operation<'c>) -> OperationRef<'c, 'v> {
        unsafe {
            let operation = operation.into_raw();

            mlirBlockInsertOwnedOperation(self.raw, position as isize, operation);

            OperationRef::from_raw(operation)
        }
    }

    fn insert_operation_after(
        &self,
        one: OperationRef<'c, 'v>,
        other: Operation<'c>,
    ) -> OperationRef<'c, 'v> {
        unsafe {
            let other = other.into_raw();

            mlirBlockInsertOwnedOperationAfter(self.raw, one.to_raw(), other);

            OperationRef::from_raw(other)
        }
    }

    fn insert_operation_before(
        &self,
        one: OperationRef<'c, 'v>,
        other: Operation<'c>,
    ) -> OperationRef<'c, 'v> {
        unsafe {
            let other = other.into_raw();

            mlirBlockInsertOwnedOperationBefore(self.raw, one.to_raw(), other);

            OperationRef::from_raw(other)
        }
    }

    fn next_in_region(&self) -> Option<BlockRef<'c, 'v>> {
        unsafe { BlockRef::from_option_raw(mlirBlockGetNextInRegion(self.raw)) }
    }
}

impl Drop for Block<'_> {
    fn drop(&mut self) {
        unsafe { mlirBlockDestroy(self.raw) };
    }
}

impl PartialEq for Block<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirBlockEqual(self.raw, other.raw) }
    }
}

impl Eq for Block<'_> {}

impl Display for Block<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirBlockPrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl Debug for Block<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        writeln!(formatter, "Block(")?;
        Display::fmt(self, formatter)?;
        write!(formatter, ")")
    }
}

/// A reference of a block.
#[derive(Clone, Copy)]
pub struct BlockRef<'c, 'a> {
    raw: MlirBlock,
    _reference: PhantomData<&'a Block<'c>>,
}

impl BlockRef<'_, '_> {
    /// Creates a block reference from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirBlock) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
    }

    /// Creates an optional block reference from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_option_raw(raw: MlirBlock) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}

impl<'c, 'v> BlockApi<'c, 'v> for BlockRef<'c, 'v> {
    fn argument(&self, index: usize) -> Result<BlockArgument<'c, 'v>, Error> {
        let block = unsafe { Block::from_raw(self.raw) };
        let result = block.argument(index);
        Block::into_raw(block);
        result
    }

    fn argument_count(&self) -> usize {
        let block = unsafe { Block::from_raw(self.raw) };
        let result = block.argument_count();
        Block::into_raw(block);
        result
    }

    fn first_operation(&self) -> Option<OperationRef<'c, 'v>> {
        let block = unsafe { Block::from_raw(self.raw) };
        let result = block.first_operation();
        Block::into_raw(block);
        result
    }

    fn first_operation_mut(&mut self) -> Option<OperationRefMut<'c, 'v>> {
        let mut block = unsafe { Block::from_raw(self.raw) };
        let result = block.first_operation_mut();
        Block::into_raw(block);
        result
    }

    fn terminator(&self) -> Option<OperationRef<'c, 'v>> {
        let block = unsafe { Block::from_raw(self.raw) };
        let result = block.terminator();
        Block::into_raw(block);
        result
    }

    fn terminator_mut(&mut self) -> Option<OperationRefMut<'c, 'v>> {
        let mut block = unsafe { Block::from_raw(self.raw) };
        let result = block.terminator_mut();
        Block::into_raw(block);
        result
    }

    fn parent_region(&self) -> Option<RegionRef<'c, 'v>> {
        let block = unsafe { Block::from_raw(self.raw) };
        let result = block.parent_region();
        Block::into_raw(block);
        result
    }

    fn parent_operation(&self) -> Option<OperationRef<'c, 'v>> {
        let block = unsafe { Block::from_raw(self.raw) };
        let result = block.parent_operation();
        Block::into_raw(block);
        result
    }

    fn add_argument(&self, r#type: Type<'c>, location: Location<'c>) -> Value<'c, 'v> {
        let block = unsafe { Block::from_raw(self.raw) };
        let result = block.add_argument(r#type, location);
        Block::into_raw(block);
        result
    }

    fn append_operation(&self, operation: Operation<'c>) -> OperationRef<'c, 'v> {
        let block = unsafe { Block::from_raw(self.raw) };
        let result = block.append_operation(operation);
        Block::into_raw(block);
        result
    }

    fn insert_operation(&self, position: usize, operation: Operation<'c>) -> OperationRef<'c, 'v> {
        let block = unsafe { Block::from_raw(self.raw) };
        let result = block.insert_operation(position, operation);
        Block::into_raw(block);
        result
    }

    fn insert_operation_after(
        &self,
        one: OperationRef<'c, 'v>,
        other: Operation<'c>,
    ) -> OperationRef<'c, 'v> {
        let block = unsafe { Block::from_raw(self.raw) };
        let result = block.insert_operation_after(one, other);
        Block::into_raw(block);
        result
    }

    fn insert_operation_before(
        &self,
        one: OperationRef<'c, 'v>,
        other: Operation<'c>,
    ) -> OperationRef<'c, 'v> {
        let block = unsafe { Block::from_raw(self.raw) };
        let result = block.insert_operation_before(one, other);
        Block::into_raw(block);
        result
    }

    fn next_in_region(&self) -> Option<BlockRef<'c, 'v>> {
        let block = unsafe { Block::from_raw(self.raw) };
        let result = block.next_in_region();
        Block::into_raw(block);
        result
    }
}

impl<'c> Deref for BlockRef<'c, '_> {
    type Target = Block<'c>;

    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}

impl PartialEq for BlockRef<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirBlockEqual(self.raw, other.raw) }
    }
}

impl Eq for BlockRef<'_, '_> {}

impl Display for BlockRef<'_, '_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self.deref(), formatter)
    }
}

impl Debug for BlockRef<'_, '_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Debug::fmt(self.deref(), formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{operation::OperationBuilder, r#type::IntegerType, Module, Region, ValueLike},
        test::create_test_context,
    };
    use pretty_assertions::assert_eq;

    #[test]
    fn new() {
        Block::new(&[]);
    }

    #[test]
    fn argument() {
        let context = create_test_context();
        let r#type = IntegerType::new(&context, 64).into();

        assert_eq!(
            Block::new(&[(r#type, Location::unknown(&context))])
                .argument(0)
                .unwrap()
                .r#type(),
            r#type
        );
    }

    #[test]
    fn argument_error() {
        assert_eq!(
            Block::new(&[]).argument(0).unwrap_err(),
            Error::PositionOutOfBounds {
                name: "block argument",
                value: "<<UNLINKED BLOCK>>\n".into(),
                index: 0,
            }
        );
    }

    #[test]
    fn argument_count() {
        assert_eq!(Block::new(&[]).argument_count(), 0);
    }

    #[test]
    fn parent_region() {
        let region = Region::new();
        let block = region.append_block(Block::new(&[]));

        assert_eq!(block.parent_region().as_deref(), Some(&region));
    }

    #[test]
    fn parent_region_none() {
        let block = Block::new(&[]);

        assert_eq!(block.parent_region(), None);
    }

    #[test]
    fn parent_operation() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));

        assert_eq!(
            module.body().parent_operation(),
            Some(module.as_operation())
        );
    }

    #[test]
    fn parent_operation_none() {
        let block = Block::new(&[]);

        assert_eq!(block.parent_operation(), None);
    }

    #[test]
    fn terminator() {
        let context = create_test_context();

        let block = Block::new(&[]);

        let operation = block.append_operation(
            OperationBuilder::new("func.return", Location::unknown(&context))
                .build()
                .unwrap(),
        );

        assert_eq!(block.terminator(), Some(operation));
    }

    #[test]
    fn terminator_none() {
        assert_eq!(Block::new(&[]).terminator(), None);
    }

    #[test]
    fn first_operation() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let block = Block::new(&[]);

        let operation = block.append_operation(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap(),
        );

        assert_eq!(block.first_operation(), Some(operation));
    }

    #[test]
    fn first_operation_none() {
        let block = Block::new(&[]);

        assert_eq!(block.first_operation(), None);
    }

    #[test]
    fn append_operation() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let block = Block::new(&[]);

        block.append_operation(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap(),
        );
    }

    #[test]
    fn insert_operation() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let block = Block::new(&[]);

        block.insert_operation(
            0,
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap(),
        );
    }

    #[test]
    fn insert_operation_after() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let block = Block::new(&[]);

        let first_operation = block.append_operation(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap(),
        );
        let second_operation = block.insert_operation_after(
            first_operation,
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap(),
        );

        assert_eq!(block.first_operation(), Some(first_operation));
        assert_eq!(
            block.first_operation().unwrap().next_in_block(),
            Some(second_operation)
        );
    }

    #[test]
    fn insert_operation_before() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let block = Block::new(&[]);

        let second_operation = block.append_operation(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap(),
        );
        let first_operation = block.insert_operation_before(
            second_operation,
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap(),
        );

        assert_eq!(block.first_operation(), Some(first_operation));
        assert_eq!(
            block.first_operation().unwrap().next_in_block(),
            Some(second_operation)
        );
    }

    #[test]
    fn next_in_region() {
        let region = Region::new();

        let first_block = region.append_block(Block::new(&[]));
        let second_block = region.append_block(Block::new(&[]));

        assert_eq!(first_block.next_in_region(), Some(second_block));
    }

    #[test]
    fn detach() {
        let region = Region::new();
        let block = region.append_block(Block::new(&[]));

        assert_eq!(
            unsafe { block.detach() }.unwrap().to_string(),
            "<<UNLINKED BLOCK>>\n"
        );
    }

    #[test]
    fn detach_detached() {
        let block = Block::new(&[]);

        assert!(unsafe { block.detach() }.is_none());
    }

    #[test]
    fn display() {
        assert_eq!(Block::new(&[]).to_string(), "<<UNLINKED BLOCK>>\n");
    }

    #[test]
    fn debug() {
        assert_eq!(
            format!("{:?}", &Block::new(&[])),
            "Block(\n<<UNLINKED BLOCK>>\n)"
        );
    }
}
