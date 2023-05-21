//! Operations and operation builders.

mod builder;
mod printing_flags;
mod result;

pub use self::{
    builder::OperationBuilder, printing_flags::OperationPrintingFlags, result::OperationResult,
};
use super::{BlockRef, Identifier, RegionRef, Value};
use crate::{
    context::{Context, ContextRef},
    utility::{print_callback, print_string_callback},
    Error,
};
use core::{
    fmt,
    mem::{forget, transmute},
};
use mlir_sys::{
    mlirOperationClone, mlirOperationDestroy, mlirOperationDump, mlirOperationEqual,
    mlirOperationGetBlock, mlirOperationGetContext, mlirOperationGetName,
    mlirOperationGetNextInBlock, mlirOperationGetNumRegions, mlirOperationGetNumResults,
    mlirOperationGetRegion, mlirOperationGetResult, mlirOperationPrint,
    mlirOperationPrintWithFlags, mlirOperationVerify, MlirOperation,
};
use std::{
    ffi::c_void,
    fmt::{Debug, Display, Formatter},
    marker::PhantomData,
    ops::Deref,
};

/// An operation.
pub struct Operation<'c> {
    raw: MlirOperation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Operation<'c> {
    /// Gets a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirOperationGetContext(self.raw)) }
    }

    /// Gets a name.
    pub fn name(&self) -> Identifier<'c> {
        unsafe { Identifier::from_raw(mlirOperationGetName(self.raw)) }
    }

    /// Gets a block.
    pub fn block(&self) -> Option<BlockRef> {
        unsafe { BlockRef::from_option_raw(mlirOperationGetBlock(self.raw)) }
    }

    /// Gets a result at a position.
    pub fn result(&self, index: usize) -> Result<OperationResult, Error> {
        unsafe {
            if index < self.result_count() {
                Ok(OperationResult::from_raw(mlirOperationGetResult(
                    self.raw,
                    index as isize,
                )))
            } else {
                Err(Error::PositionOutOfBounds {
                    name: "operation result",
                    value: self.to_string(),
                    index,
                })
            }
        }
    }

    /// Gets a number of results.
    pub fn result_count(&self) -> usize {
        unsafe { mlirOperationGetNumResults(self.raw) as usize }
    }

    /// Gets a region at a position.
    pub fn region(&self, index: usize) -> Result<RegionRef, Error> {
        unsafe {
            if index < self.region_count() {
                Ok(RegionRef::from_raw(mlirOperationGetRegion(
                    self.raw,
                    index as isize,
                )))
            } else {
                Err(Error::PositionOutOfBounds {
                    name: "region",
                    value: self.to_string(),
                    index,
                })
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

    /// Prints an operation with flags.
    pub fn to_string_with_flags(&self, flags: OperationPrintingFlags) -> Result<String, Error> {
        let mut data = (String::new(), Ok::<_, Error>(()));

        unsafe {
            mlirOperationPrintWithFlags(
                self.raw,
                flags.to_raw(),
                Some(print_string_callback),
                &mut data as *mut _ as *mut _,
            );
        }

        data.1?;

        Ok(data.0)
    }

    /// Creates an operation from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirOperation) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Converts an operation into a raw object.
    pub fn into_raw(self) -> MlirOperation {
        let operation = self.raw;

        forget(self);

        operation
    }
}

impl<'c> Clone for Operation<'c> {
    fn clone(&self) -> Self {
        unsafe { Self::from_raw(mlirOperationClone(self.raw)) }
    }
}

impl<'c> Drop for Operation<'c> {
    fn drop(&mut self) {
        unsafe { mlirOperationDestroy(self.raw) };
    }
}

impl<'c> PartialEq for Operation<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirOperationEqual(self.raw, other.raw) }
    }
}

impl<'c> Eq for Operation<'c> {}

impl<'a> Display for Operation<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirOperationPrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl<'c> Debug for Operation<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        writeln!(formatter, "Operation(")?;
        Display::fmt(self, formatter)?;
        write!(formatter, ")")
    }
}

/// A reference to an operation.
// TODO Should we split context lifetimes? Or, is it transitively proven that
// 'c > 'a?
#[derive(Clone, Copy)]
pub struct OperationRef<'a> {
    raw: MlirOperation,
    _reference: PhantomData<&'a Operation<'a>>,
}

impl<'a> OperationRef<'a> {
    /// Gets a result at a position.
    pub fn result(self, index: usize) -> Result<OperationResult<'a>, Error> {
        // As we can't deref OperationRef<'a> into `&'a Operation`, we forcibly cast its
        // lifetime here to extend it from the lifetime of `ObjectRef<'a>` itself into
        // `'a`.
        unsafe { transmute(self.deref().result(index)) }
    }

    /// Converts an operation reference into a raw object.
    pub fn to_raw(self) -> MlirOperation {
        self.raw
    }

    /// Creates an operation reference from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirOperation) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
    }

    /// Creates an optional operation reference from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_option_raw(raw: MlirOperation) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}

impl<'a> Deref for OperationRef<'a> {
    type Target = Operation<'a>;

    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
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
        Display::fmt(self.deref(), formatter)
    }
}

impl<'a> Debug for OperationRef<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Debug::fmt(self.deref(), formatter)
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
        OperationBuilder::new("foo", Location::unknown(&Context::new())).build();
    }

    #[test]
    fn name() {
        let context = Context::new();

        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context),)
                .build()
                .name(),
            Identifier::new(&context, "foo")
        );
    }

    #[test]
    fn block() {
        let block = Block::new(&[]);
        let operation = block.append_operation(
            OperationBuilder::new("foo", Location::unknown(&Context::new())).build(),
        );

        assert_eq!(operation.block().as_deref(), Some(&block));
    }

    #[test]
    fn block_none() {
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&Context::new()))
                .build()
                .block(),
            None
        );
    }

    #[test]
    fn result_error() {
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&Context::new()))
                .build()
                .result(0)
                .unwrap_err(),
            Error::PositionOutOfBounds {
                name: "operation result",
                value: "\"foo\"() : () -> ()\n".into(),
                index: 0
            }
        );
    }

    #[test]
    fn region_none() {
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&Context::new()),)
                .build()
                .region(0),
            Err(Error::PositionOutOfBounds {
                name: "region",
                value: "\"foo\"() : () -> ()\n".into(),
                index: 0
            })
        );
    }

    #[test]
    fn clone() {
        let context = Context::new();
        let operation = OperationBuilder::new("foo", Location::unknown(&context)).build();

        let _ = operation.clone();
    }

    #[test]
    fn display() {
        let context = Context::new();

        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context),)
                .build()
                .to_string(),
            "\"foo\"() : () -> ()\n"
        );
    }

    #[test]
    fn debug() {
        let context = Context::new();

        assert_eq!(
            format!(
                "{:?}",
                OperationBuilder::new("foo", Location::unknown(&context)).build()
            ),
            "Operation(\n\"foo\"() : () -> ()\n)"
        );
    }

    #[test]
    fn to_string_with_flags() {
        let context = Context::new();

        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .to_string_with_flags(
                    OperationPrintingFlags::new()
                        .elide_large_elements_attributes(100)
                        .enable_debug_info(true, true)
                        .print_generic_operation_form()
                        .use_local_scope()
                ),
            Ok("\"foo\"() : () -> () [unknown]".into())
        );
    }
}
