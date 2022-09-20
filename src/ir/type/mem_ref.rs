use super::TypeLike;
use crate::{
    ir::{Attribute, Type},
    Error,
};
use mlir_sys::{
    mlirMemRefTypeGet, mlirMemRefTypeGetLayout, mlirMemRefTypeGetMemorySpace, MlirType,
};
use std::fmt::{self, Display, Formatter};

/// A mem-ref type.
#[derive(Clone, Copy, Debug)]
pub struct MemRef<'c> {
    r#type: Type<'c>,
}

impl<'c> MemRef<'c> {
    /// Creates a mem-ref type.
    pub fn new(
        r#type: Type<'c>,
        dimensions: &[u64],
        layout: Attribute<'c>,
        memory_space: Attribute<'c>,
    ) -> Self {
        Self {
            r#type: unsafe {
                Type::from_raw(mlirMemRefTypeGet(
                    r#type.to_raw(),
                    dimensions.len() as isize,
                    dimensions.as_ptr() as *const i64,
                    layout.to_raw(),
                    memory_space.to_raw(),
                ))
            },
        }
    }

    /// Gets a layout.
    pub fn layout(&self) -> Attribute<'c> {
        unsafe { Attribute::from_raw(mlirMemRefTypeGetLayout(self.r#type.to_raw())) }
    }

    /// Gets a memory space.
    pub fn memory_space(&self) -> Attribute<'c> {
        unsafe { Attribute::from_raw(mlirMemRefTypeGetMemorySpace(self.r#type.to_raw())) }
    }
}

impl<'c> TypeLike<'c> for MemRef<'c> {
    fn to_raw(&self) -> MlirType {
        self.r#type.to_raw()
    }
}

impl<'c> Display for MemRef<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Type::from(*self).fmt(formatter)
    }
}

impl<'c> TryFrom<Type<'c>> for MemRef<'c> {
    type Error = Error;

    fn try_from(r#type: Type<'c>) -> Result<Self, Self::Error> {
        if r#type.is_mem_ref() {
            Ok(Self { r#type })
        } else {
            Err(Error::MemRefExpected(r#type.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn new() {
        let context = Context::new();

        assert_eq!(
            Type::from(MemRef::new(&context, &[])),
            Type::parse(&context, "memref<>").unwrap()
        );
    }
}
