use super::TypeLike;
use crate::{
    ir::{affine_map::AffineMap, Attribute, Location, Type},
    Error,
};
use mlir_sys::{
    mlirMemRefTypeGet, mlirMemRefTypeGetAffineMap, mlirMemRefTypeGetChecked,
    mlirMemRefTypeGetLayout, mlirMemRefTypeGetMemorySpace, MlirType,
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
        unsafe {
            Self::from_raw(mlirMemRefTypeGet(
                r#type.to_raw(),
                dimensions.len() as isize,
                dimensions.as_ptr() as *const i64,
                layout.to_raw(),
                memory_space.to_raw(),
            ))
        }
    }

    /// Creates a mem-ref type with diagnostics.
    pub fn checked(
        location: Location<'c>,
        r#type: Type<'c>,
        dimensions: &[u64],
        layout: Attribute<'c>,
        memory_space: Attribute<'c>,
    ) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirMemRefTypeGetChecked(
                location.to_raw(),
                r#type.to_raw(),
                dimensions.len() as isize,
                dimensions.as_ptr() as *const i64,
                layout.to_raw(),
                memory_space.to_raw(),
            ))
        }
    }

    /// Gets a layout.
    pub fn layout(&self) -> Attribute<'c> {
        unsafe { Attribute::from_raw(mlirMemRefTypeGetLayout(self.r#type.to_raw())) }
    }

    /// Gets an affine map.
    pub fn affine_map(&self) -> AffineMap<'c> {
        unsafe { AffineMap::from_raw(mlirMemRefTypeGetAffineMap(self.r#type.to_raw())) }
    }

    /// Gets a memory space.
    pub fn memory_space(&self) -> Attribute<'c> {
        unsafe { Attribute::from_raw(mlirMemRefTypeGetMemorySpace(self.r#type.to_raw())) }
    }

    unsafe fn from_raw(raw: MlirType) -> Self {
        Self {
            r#type: Type::from_raw(raw),
        }
    }

    unsafe fn from_option_raw(raw: MlirType) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
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
            Type::from(MemRef::new(
                Type::integer(&context, 42),
                &[42],
                Attribute::null(),
                Attribute::null(),
            )),
            Type::parse(&context, "memref<42xi42>").unwrap()
        );
    }

    #[test]
    fn layout() {
        let context = Context::new();

        assert_eq!(
            MemRef::new(
                Type::integer(&context, 42),
                &[42, 42],
                Attribute::null(),
                Attribute::null(),
            )
            .layout(),
            Attribute::parse(&context, "affine_map<(d0, d1) -> (d0, d1)>").unwrap(),
        );
    }

    #[test]
    fn affine_map() {
        let context = Context::new();

        assert_eq!(
            MemRef::new(
                Type::integer(&context, 42),
                &[42, 42],
                Attribute::null(),
                Attribute::null(),
            )
            .affine_map()
            .to_string(),
            "(d0, d1) -> (d0, d1)"
        );
    }

    #[test]
    fn memory_space() {
        let context = Context::new();

        assert_eq!(
            MemRef::new(
                Type::integer(&context, 42),
                &[42, 42],
                Attribute::null(),
                Attribute::null(),
            )
            .memory_space(),
            Attribute::null(),
        );
    }
}
