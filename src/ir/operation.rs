//! Operations and operation builders.

mod builder;

pub use self::builder::Builder;
use super::{BlockRef, Identifier, OperationResult, RegionRef, Value};
use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
    Error,
};
use core::fmt;
use mlir_sys::{
    mlirOperationClone, mlirOperationDestroy, mlirOperationDump, mlirOperationEqual,
    mlirOperationGetBlock, mlirOperationGetContext, mlirOperationGetName,
    mlirOperationGetNextInBlock, mlirOperationGetNumRegions, mlirOperationGetNumResults,
    mlirOperationGetRegion, mlirOperationGetResult, mlirOperationPrint, mlirOperationVerify,
    MlirOperation, MlirStringRef,
};
use std::{
    ffi::c_void,
    fmt::{Display, Formatter},
    marker::PhantomData,
    mem::forget,
    ops::Deref,
};

/// An operation.
#[derive(Debug)]
pub struct Operation<'c> {
    r#ref: OperationRef<'static>,
    _context: PhantomData<&'c Context>,
}

impl<'c> Operation<'c> {
    pub(crate) unsafe fn from_raw(raw: MlirOperation) -> Self {
        Self {
            r#ref: OperationRef::from_raw(raw),
            _context: Default::default(),
        }
    }

    pub(crate) unsafe fn into_raw(self) -> MlirOperation {
        let operation = self.raw;

        forget(self);

        operation
    }
}

impl<'c> Drop for Operation<'c> {
    fn drop(&mut self) {
        unsafe { mlirOperationDestroy(self.raw) };
    }
}

impl<'c> PartialEq for Operation<'c> {
    fn eq(&self, other: &Self) -> bool {
        self.r#ref == other.r#ref
    }
}

impl<'c> Eq for Operation<'c> {}

impl<'c> Deref for Operation<'c> {
    type Target = OperationRef<'static>;

    fn deref(&self) -> &Self::Target {
        &self.r#ref
    }
}

/// A reference to an operation.
// TODO Should we split context lifetimes? Or, is it transitively proven that
// 'c > 'a?
#[derive(Clone, Copy, Debug)]
pub struct OperationRef<'a> {
    raw: MlirOperation,
    _reference: PhantomData<&'a Operation<'a>>,
}

impl<'a> OperationRef<'a> {
    /// Gets a context.
    pub fn context(&self) -> ContextRef {
        unsafe { ContextRef::from_raw(mlirOperationGetContext(self.raw)) }
    }

    /// Gets a name.
    pub fn name(&self) -> Identifier {
        unsafe { Identifier::from_raw(mlirOperationGetName(self.raw)) }
    }

    /// Gets a block.
    pub fn block(&self) -> Option<BlockRef> {
        unsafe { BlockRef::from_option_raw(mlirOperationGetBlock(self.raw)) }
    }

    /// Gets a result at a position.
    pub fn result(&self, position: usize) -> Result<OperationResult, Error> {
        unsafe {
            if position < self.result_count() as usize {
                Ok(OperationResult::from_value(Value::from_raw(
                    mlirOperationGetResult(self.raw, position as isize),
                )))
            } else {
                Err(Error::OperationResultPosition(self.to_string(), position))
            }
        }
    }

    /// Gets a number of results.
    pub fn result_count(&self) -> usize {
        unsafe { mlirOperationGetNumResults(self.raw) as usize }
    }

    /// Gets a result at an index.
    pub fn region(&self, index: usize) -> Option<RegionRef> {
        unsafe {
            if index < self.region_count() as usize {
                Some(RegionRef::from_raw(mlirOperationGetRegion(
                    self.raw,
                    index as isize,
                )))
            } else {
                None
            }
        }
    }

    /// Gets a number of regions.
    pub fn region_count(&self) -> usize {
        unsafe { mlirOperationGetNumRegions(self.raw) as usize }
    }

    /// Gets the next operation in the same block.
    pub fn next_in_block(&self) -> Option<OperationRef> {
        unsafe {
            let operation = mlirOperationGetNextInBlock(self.raw);

            if operation.ptr.is_null() {
                None
            } else {
                Some(OperationRef::from_raw(operation))
            }
        }
    }

    /// Verifies an operation.
    pub fn verify(&self) -> bool {
        unsafe { mlirOperationVerify(self.raw) }
    }

    /// Dumps an operation.
    pub fn dump(&self) {
        unsafe { mlirOperationDump(self.raw) }
    }

    /// Clones an operation.
    pub fn to_owned(&self) -> Operation {
        unsafe { Operation::from_raw(mlirOperationClone(self.raw)) }
    }

    pub(crate) unsafe fn to_raw(self) -> MlirOperation {
        self.raw
    }

    pub(crate) unsafe fn from_raw(raw: MlirOperation) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
    }

    pub(crate) unsafe fn from_option_raw(raw: MlirOperation) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}

impl<'a> PartialEq for OperationRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirOperationEqual(self.raw, other.raw) }
    }
}

impl<'a> Eq for OperationRef<'a> {}

impl<'a> Display for OperationRef<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe extern "C" fn callback(string: MlirStringRef, data: *mut c_void) {
            let data = &mut *(data as *mut (&mut Formatter, fmt::Result));
            let result = write!(data.0, "{}", StringRef::from_raw(string).as_str());

            if data.1.is_ok() {
                data.1 = result;
            }
        }

        unsafe {
            mlirOperationPrint(self.raw, Some(callback), &mut data as *mut _ as *mut c_void);
        }

        data.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        ir::{Block, Location},
    };
    use pretty_assertions::assert_eq;

    #[test]
    fn new() {
        Builder::new("foo", Location::unknown(&Context::new())).build();
    }

    #[test]
    fn name() {
        let context = Context::new();

        assert_eq!(
            Builder::new("foo", Location::unknown(&context),)
                .build()
                .name(),
            Identifier::new(&context, "foo")
        );
    }

    #[test]
    fn block() {
        let block = Block::new(&[]);
        let operation =
            block.append_operation(Builder::new("foo", Location::unknown(&Context::new())).build());

        assert_eq!(operation.block(), Some(*block));
    }

    #[test]
    fn block_none() {
        assert_eq!(
            Builder::new("foo", Location::unknown(&Context::new()))
                .build()
                .block(),
            None
        );
    }

    #[test]
    fn result_none() {
        assert_eq!(
            Builder::new("foo", Location::unknown(&Context::new()))
                .build()
                .result(0)
                .unwrap_err(),
            Error::OperationResultPosition("\"foo\"".into(), 0)
        );
    }

    #[test]
    fn region_none() {
        assert!(Builder::new("foo", Location::unknown(&Context::new()),)
            .build()
            .region(0)
            .is_none());
    }

    #[test]
    fn to_owned() {
        let context = Context::new();
        let operation = Builder::new("foo", Location::unknown(&context)).build();

        operation.to_owned();
    }
}
