//! Operations and operation builders.

mod builder;
mod printing_flags;
mod result;

pub use self::{
    builder::OperationBuilder, printing_flags::OperationPrintingFlags, result::OperationResult,
};
use super::{Attribute, AttributeLike, BlockRef, Identifier, RegionRef, Value};
use crate::{
    context::{Context, ContextRef},
    utility::{print_callback, print_string_callback},
    Error, StringRef,
};
use core::{
    fmt,
    mem::{forget, transmute},
};
use mlir_sys::{
    mlirOperationClone, mlirOperationDestroy, mlirOperationDump, mlirOperationEqual,
    mlirOperationGetAttribute, mlirOperationGetAttributeByName, mlirOperationGetBlock,
    mlirOperationGetContext, mlirOperationGetName, mlirOperationGetNextInBlock,
    mlirOperationGetNumAttributes, mlirOperationGetNumOperands, mlirOperationGetNumRegions,
    mlirOperationGetNumResults, mlirOperationGetNumSuccessors, mlirOperationGetOperand,
    mlirOperationGetRegion, mlirOperationGetResult, mlirOperationGetSuccessor, mlirOperationPrint,
    mlirOperationPrintWithFlags, mlirOperationRemoveAttributeByName,
    mlirOperationSetAttributeByName, mlirOperationVerify, MlirOperation,
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
    // TODO Store lifetime of block in operations, or create another type like
    // `AppendedOperationRef`?
    pub fn block(&self) -> Option<BlockRef<'c, '_>> {
        unsafe { BlockRef::from_option_raw(mlirOperationGetBlock(self.raw)) }
    }

    /// Gets the number of operands.
    pub fn operand_count(&self) -> usize {
        unsafe { mlirOperationGetNumOperands(self.raw) as usize }
    }

    /// Gets the operand at a position.
    pub fn operand(&self, index: usize) -> Result<Value<'c, '_>, Error> {
        if index < self.operand_count() {
            Ok(unsafe { Value::from_raw(mlirOperationGetOperand(self.raw, index as isize)) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "operation operand",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Gets all operands.
    pub fn operands(&self) -> impl Iterator<Item = Value<'c, '_>> {
        (0..self.operand_count()).map(|index| self.operand(index).expect("valid operand index"))
    }

    /// Gets the number of results.
    pub fn result_count(&self) -> usize {
        unsafe { mlirOperationGetNumResults(self.raw) as usize }
    }

    /// Gets a result at a position.
    pub fn result(&self, index: usize) -> Result<OperationResult<'c, '_>, Error> {
        if index < self.result_count() {
            Ok(unsafe {
                OperationResult::from_raw(mlirOperationGetResult(self.raw, index as isize))
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "operation result",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Gets all results.
    pub fn results(&self) -> impl Iterator<Item = OperationResult<'c, '_>> {
        (0..self.result_count()).map(|index| self.result(index).expect("valid result index"))
    }

    /// Gets the number of regions.
    pub fn region_count(&self) -> usize {
        unsafe { mlirOperationGetNumRegions(self.raw) as usize }
    }

    /// Gets a region at a position.
    pub fn region(&self, index: usize) -> Result<RegionRef<'c, '_>, Error> {
        if index < self.region_count() {
            Ok(unsafe { RegionRef::from_raw(mlirOperationGetRegion(self.raw, index as isize)) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "region",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Gets all regions.
    pub fn regions(&self) -> impl Iterator<Item = RegionRef<'c, '_>> {
        (0..self.region_count()).map(|index| self.region(index).expect("valid result index"))
    }

    /// Gets the number of successors.
    pub fn successor_count(&self) -> usize {
        unsafe { mlirOperationGetNumSuccessors(self.raw) as usize }
    }

    /// Gets a successor at a position.
    pub fn successor(&self, index: usize) -> Result<BlockRef<'c, '_>, Error> {
        if index < self.successor_count() {
            Ok(unsafe { BlockRef::from_raw(mlirOperationGetSuccessor(self.raw, index as isize)) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "successor",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Gets all successors.
    pub fn successors(&self) -> impl Iterator<Item = BlockRef<'c, '_>> {
        (0..self.successor_count())
            .map(|index| self.successor(index).expect("valid successor index"))
    }

    /// Gets the number of attributes.
    pub fn attribute_count(&self) -> usize {
        unsafe { mlirOperationGetNumAttributes(self.raw) as usize }
    }

    /// Gets a attribute at a position.
    pub fn attribute_at(&self, index: usize) -> Result<(Identifier<'c>, Attribute<'c>), Error> {
        if index < self.attribute_count() {
            unsafe {
                let named_attribute = mlirOperationGetAttribute(self.raw, index as isize);
                Ok((
                    Identifier::from_raw(named_attribute.name),
                    Attribute::from_raw(named_attribute.attribute),
                ))
            }
        } else {
            Err(Error::PositionOutOfBounds {
                name: "attribute",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Gets all attributes.
    pub fn attributes(&self) -> impl Iterator<Item = (Identifier<'c>, Attribute<'c>)> + '_ {
        (0..self.attribute_count())
            .map(|index| self.attribute_at(index).expect("valid attribute index"))
    }

    /// Gets a attribute with the given name.
    pub fn attribute(&self, name: &str) -> Result<Attribute<'c>, Error> {
        unsafe {
            Attribute::from_option_raw(mlirOperationGetAttributeByName(
                self.raw,
                StringRef::new(name).to_raw(),
            ))
        }
        .ok_or(Error::AttributeNotFound(name.into()))
    }

    /// Checks if the operation has a attribute with the given name.
    pub fn has_attribute(&self, name: &str) -> bool {
        self.attribute(name).is_ok()
    }

    /// Sets the attribute with the given name to the given attribute.
    pub fn set_attribute(&mut self, name: &str, attribute: Attribute<'c>) {
        unsafe {
            mlirOperationSetAttributeByName(
                self.raw,
                StringRef::new(name).to_raw(),
                attribute.to_raw(),
            )
        }
    }

    /// Removes the attribute with the given name.
    pub fn remove_attribute(&mut self, name: &str) -> Result<(), Error> {
        unsafe { mlirOperationRemoveAttributeByName(self.raw, StringRef::new(name).to_raw()) }
            .then_some(())
            .ok_or(Error::AttributeNotFound(name.into()))
    }

    /// Gets the next operation in the same block.
    pub fn next_in_block(&self) -> Option<OperationRef<'c, '_>> {
        unsafe { OperationRef::from_option_raw(mlirOperationGetNextInBlock(self.raw)) }
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

    /// Creates an optional operation from a raw object.
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
#[derive(Clone, Copy)]
pub struct OperationRef<'c, 'a> {
    raw: MlirOperation,
    _reference: PhantomData<&'a Operation<'c>>,
}

impl<'c, 'a> OperationRef<'c, 'a> {
    /// Gets a result at a position.
    pub fn result(self, index: usize) -> Result<OperationResult<'c, 'a>, Error> {
        unsafe { self.to_ref() }.result(index)
    }

    /// Gets an operation.
    ///
    /// This function is different from `deref` because the correct lifetime is
    /// kept for the return type.
    ///
    /// # Safety
    ///
    /// The returned reference is safe to use only in the lifetime scope of the
    /// operation reference.
    pub unsafe fn to_ref(&self) -> &'a Operation<'c> {
        // As we can't deref OperationRef<'a> into `&'a Operation`, we forcibly cast its
        // lifetime here to extend it from the lifetime of `ObjectRef<'a>` itself into
        // `'a`.
        transmute(self)
    }

    /// Converts an operation reference into a raw object.
    pub const fn to_raw(self) -> MlirOperation {
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

impl<'c, 'a> Deref for OperationRef<'c, 'a> {
    type Target = Operation<'a>;

    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}

impl<'c, 'a> PartialEq for OperationRef<'c, 'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirOperationEqual(self.raw, other.raw) }
    }
}

impl<'c, 'a> Eq for OperationRef<'c, 'a> {}

impl<'c, 'a> Display for OperationRef<'c, 'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self.deref(), formatter)
    }
}

impl<'c, 'a> Debug for OperationRef<'c, 'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Debug::fmt(self.deref(), formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        ir::{attribute::StringAttribute, Block, Location, Region, Type},
        test::create_test_context,
    };
    use pretty_assertions::assert_eq;

    #[test]
    fn new() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        OperationBuilder::new("foo", Location::unknown(&context))
            .build()
            .unwrap();
    }

    #[test]
    fn name() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);

        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context),)
                .build()
                .unwrap()
                .name(),
            Identifier::new(&context, "foo")
        );
    }

    #[test]
    fn block() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let block = Block::new(&[]);
        let operation = block.append_operation(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap(),
        );

        assert_eq!(operation.block().as_deref(), Some(&block));
    }

    #[test]
    fn block_none() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap()
                .block(),
            None
        );
    }

    #[test]
    fn result_error() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap()
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
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context),)
                .build()
                .unwrap()
                .region(0),
            Err(Error::PositionOutOfBounds {
                name: "region",
                value: "\"foo\"() : () -> ()\n".into(),
                index: 0
            })
        );
    }

    #[test]
    fn operands() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, location)]);
        let argument: Value = block.argument(0).unwrap().into();

        let operands = vec![argument, argument, argument];
        let operation = OperationBuilder::new("foo", Location::unknown(&context))
            .add_operands(&operands)
            .build()
            .unwrap();

        assert_eq!(
            operation.operands().skip(1).collect::<Vec<_>>(),
            vec![argument, argument]
        );
    }

    #[test]
    fn regions() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let operation = OperationBuilder::new("foo", Location::unknown(&context))
            .add_regions([Region::new()])
            .build()
            .unwrap();

        assert_eq!(
            operation.regions().collect::<Vec<_>>(),
            vec![operation.region(0).unwrap()]
        );
    }

    #[test]
    fn attribute() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let mut operation = OperationBuilder::new("foo", Location::unknown(&context))
            .add_attributes(&[(
                Identifier::new(&context, "foo"),
                StringAttribute::new(&context, "bar").into(),
            )])
            .build()
            .unwrap();
        assert!(operation.has_attribute("foo"));
        assert_eq!(
            operation.attribute("foo").map(|a| a.to_string()),
            Ok("\"bar\"".into())
        );
        assert!(operation.remove_attribute("foo").is_ok());
        assert!(operation.remove_attribute("foo").is_err());
        operation.set_attribute("foo", StringAttribute::new(&context, "foo").into());
        assert_eq!(
            operation.attribute("foo").map(|a| a.to_string()),
            Ok("\"foo\"".into())
        );
        assert_eq!(
            operation.attributes().next(),
            Some((
                Identifier::new(&context, "foo"),
                StringAttribute::new(&context, "foo").into()
            ))
        )
    }

    #[test]
    fn clone() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let operation = OperationBuilder::new("foo", Location::unknown(&context))
            .build()
            .unwrap();

        let _ = operation.clone();
    }

    #[test]
    fn display() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context),)
                .build()
                .unwrap()
                .to_string(),
            "\"foo\"() : () -> ()\n"
        );
    }

    #[test]
    fn debug() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        assert_eq!(
            format!(
                "{:?}",
                OperationBuilder::new("foo", Location::unknown(&context))
                    .build()
                    .unwrap()
            ),
            "Operation(\n\"foo\"() : () -> ()\n)"
        );
    }

    #[test]
    fn to_string_with_flags() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap()
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
