use crate::{
    block::BlockRef,
    context::{Context, ContextRef},
    location::Location,
    operation::{Operation, OperationRef},
    string_ref::StringRef,
};
use mlir_sys::{
    mlirModuleCreateEmpty, mlirModuleCreateParse, mlirModuleDestroy, mlirModuleFromOperation,
    mlirModuleGetBody, mlirModuleGetContext, mlirModuleGetOperation, MlirModule,
};
use std::marker::PhantomData;

#[derive(Debug)]
pub struct Module<'c> {
    raw: MlirModule,
    _context: PhantomData<&'c Context>,
}

impl<'c> Module<'c> {
    pub fn new(location: Location) -> Self {
        unsafe { Self::from_raw(mlirModuleCreateEmpty(location.to_raw())) }
    }

    pub fn parse(context: &Context, source: &str) -> Option<Self> {
        // TODO Should we allocate StringRef locally because sources can be big?
        unsafe {
            Self::from_option_raw(mlirModuleCreateParse(
                context.to_raw(),
                StringRef::from(source).to_raw(),
            ))
        }
    }

    pub fn as_operation(&self) -> OperationRef {
        unsafe { OperationRef::from_raw(mlirModuleGetOperation(self.raw)) }
    }

    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirModuleGetContext(self.raw)) }
    }

    pub fn body(&self) -> BlockRef {
        unsafe { BlockRef::from_raw(mlirModuleGetBody(self.raw)) }
    }

    pub fn from_operation(operation: Operation) -> Option<Self> {
        unsafe { Self::from_option_raw(mlirModuleFromOperation(operation.into_raw())) }
    }

    unsafe fn from_raw(raw: MlirModule) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    unsafe fn from_option_raw(raw: MlirModule) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirModule {
        self.raw
    }
}

impl<'c> Drop for Module<'c> {
    fn drop(&mut self) {
        unsafe { mlirModuleDestroy(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{block::Block, operation_state::OperationState, region::Region};

    #[test]
    fn new() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42));
    }

    #[test]
    fn context() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42)).context();
    }

    #[test]
    fn parse() {
        assert!(Module::parse(&Context::new(), "module{}").is_some());
    }

    #[test]
    fn parse_none() {
        assert!(Module::parse(&Context::new(), "module{").is_none());
    }

    #[test]
    fn from_operation() {
        let context = Context::new();

        let region = Region::new();
        region.append_block(Block::new(&[]));

        let module = Module::from_operation(Operation::new(
            OperationState::new("builtin.module", Location::unknown(&context))
                .add_regions(vec![region]),
        ))
        .unwrap();

        assert!(module.as_operation().verify());
        assert_eq!(module.as_operation().to_string(), "module {\n}\n")
    }

    #[test]
    fn from_operation_fail() {
        let context = Context::new();

        assert!(Module::from_operation(Operation::new(OperationState::new(
            "func.func",
            Location::unknown(&context),
        )))
        .is_none());
    }
}
