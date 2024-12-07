use super::arith::ArithBlockExt;
use super::builtin::BuiltinBlockExt;
use crate::{
    dialect::{llvm::r#type::pointer, ods},
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute,
        },
        r#type::IntegerType,
        Attribute, Block, Location, Type, Value, ValueLike,
    },
    Context, Error,
};

/// An index for an `llvm.getelementptr` instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GepIndex<'c, 'a> {
    /// A compile time known index.
    Const(i32),
    /// A runtime value index.
    Value(Value<'c, 'a>),
}

/// A block extension for an `llvm` dialect.
pub trait LlvmBlockExt<'c>: BuiltinBlockExt<'c> + ArithBlockExt<'c> {
    /// Creates an `llvm.extractvalue` operation.
    fn extract_value(
        &self,
        context: &'c Context,
        location: Location<'c>,
        container: Value<'c, '_>,
        value_type: Type<'c>,
        index: usize,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates an `llvm.insertvalue` operation.
    fn insert_value(
        &self,
        context: &'c Context,
        location: Location<'c>,
        container: Value<'c, '_>,
        value: Value<'c, '_>,
        index: usize,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates an `llvm.insertvalue` operation that insert multiple elements into an aggregate
    /// from the first index.
    fn insert_values<'block>(
        &'block self,
        context: &'c Context,
        location: Location<'c>,
        container: Value<'c, 'block>,
        values: &[Value<'c, 'block>],
    ) -> Result<Value<'c, 'block>, Error>;

    /// Creates an `llvm.load` operation.
    fn load(
        &self,
        context: &'c Context,
        location: Location<'c>,
        addr: Value<'c, '_>,
        value_type: Type<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates an `llvm.alloca` operation.
    fn alloca(
        &self,
        context: &'c Context,
        location: Location<'c>,
        element_type: Type<'c>,
        element_count: Value<'c, '_>,
        align: usize,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates an `llvm.alloca` operation that allocates one element.
    fn alloca1(
        &self,
        context: &'c Context,
        location: Location<'c>,
        r#type: Type<'c>,
        align: usize,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates an `llvm.alloca` operation that allocates one element of the given size of an integer.
    fn alloca_int(
        &self,
        context: &'c Context,
        location: Location<'c>,
        bits: u32,
        align: usize,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates an `llvm.store` operation.
    fn store(
        &self,
        context: &'c Context,
        location: Location<'c>,
        pointer: Value<'c, '_>,
        value: Value<'c, '_>,
    ) -> Result<(), Error>;

    /// Creates a memcpy operation.
    fn memcpy(
        &self,
        context: &'c Context,
        location: Location<'c>,
        src: Value<'c, '_>,
        dst: Value<'c, '_>,
        len_bytes: Value<'c, '_>,
    );

    /// Creates an `llvm.getelementptr` operation.
    ///
    /// This method allows combining both compile time indexes and runtime value indexes.
    fn gep(
        &self,
        context: &'c Context,
        location: Location<'c>,
        pointer: Value<'c, '_>,
        indexes: &[GepIndex<'c, '_>],
        element_type: Type<'c>,
    ) -> Result<Value<'c, '_>, Error>;
}

impl<'c> LlvmBlockExt<'c> for Block<'c> {
    #[inline]
    fn extract_value(
        &self,
        context: &'c Context,
        location: Location<'c>,
        container: Value<'c, '_>,
        value_type: Type<'c>,
        index: usize,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(
            ods::llvm::extractvalue(
                context,
                value_type,
                container,
                DenseI64ArrayAttribute::new(context, &[index.try_into().unwrap()]).into(),
                location,
            )
            .into(),
        )
    }

    #[inline]
    fn insert_value(
        &self,
        context: &'c Context,
        location: Location<'c>,
        container: Value<'c, '_>,
        value: Value<'c, '_>,
        index: usize,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(
            ods::llvm::insertvalue(
                context,
                container.r#type(),
                container,
                value,
                DenseI64ArrayAttribute::new(context, &[index.try_into().unwrap()]).into(),
                location,
            )
            .into(),
        )
    }

    #[inline]
    fn insert_values<'block>(
        &'block self,
        context: &'c Context,
        location: Location<'c>,
        mut container: Value<'c, 'block>,
        values: &[Value<'c, 'block>],
    ) -> Result<Value<'c, 'block>, Error> {
        for (i, value) in values.iter().enumerate() {
            container = self.insert_value(context, location, container, *value, i)?;
        }
        Ok(container)
    }

    #[inline]
    fn store(
        &self,
        context: &'c Context,
        location: Location<'c>,
        addr: Value<'c, '_>,
        value: Value<'c, '_>,
    ) -> Result<(), Error> {
        self.append_operation(ods::llvm::store(context, value, addr, location).into());
        Ok(())
    }

    #[inline]
    fn load(
        &self,
        context: &'c Context,
        location: Location<'c>,
        addr: Value<'c, '_>,
        value_type: Type<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(ods::llvm::load(context, value_type, addr, location).into())
    }

    #[inline]
    fn memcpy(
        &self,
        context: &'c Context,
        location: Location<'c>,
        src: Value<'c, '_>,
        dst: Value<'c, '_>,
        len_bytes: Value<'c, '_>,
    ) {
        self.append_operation(
            ods::llvm::intr_memcpy(
                context,
                dst,
                src,
                len_bytes,
                IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                location,
            )
            .into(),
        );
    }

    #[inline]
    fn alloca(
        &self,
        context: &'c Context,
        location: Location<'c>,
        element_type: Type<'c>,
        element_count: Value<'c, '_>,
        align: usize,
    ) -> Result<Value<'c, '_>, Error> {
        let mut op = ods::llvm::alloca(
            context,
            pointer(context, 0),
            element_count,
            TypeAttribute::new(element_type),
            location,
        );

        op.set_elem_type(TypeAttribute::new(element_type));
        op.set_alignment(IntegerAttribute::new(
            IntegerType::new(context, 64).into(),
            align.try_into().unwrap(),
        ));

        self.append_op_result(op.into())
    }

    #[inline]
    fn alloca1(
        &self,
        context: &'c Context,
        location: Location<'c>,
        r#type: Type<'c>,
        align: usize,
    ) -> Result<Value<'c, '_>, Error> {
        self.alloca(
            context,
            location,
            r#type,
            self.const_int(context, location, 1, 64)?,
            align,
        )
    }

    #[inline]
    fn alloca_int(
        &self,
        context: &'c Context,
        location: Location<'c>,
        bits: u32,
        align: usize,
    ) -> Result<Value<'c, '_>, Error> {
        self.alloca1(
            context,
            location,
            IntegerType::new(context, bits).into(),
            align,
        )
    }

    #[inline]
    fn gep(
        &self,
        context: &'c Context,
        location: Location<'c>,
        pointer: Value<'c, '_>,
        indexes: &[GepIndex<'c, '_>],
        element_type: Type<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        let mut dynamic_indices = Vec::with_capacity(indexes.len());
        let mut raw_constant_indices = Vec::with_capacity(indexes.len());

        for index in indexes {
            match index {
                GepIndex::Const(idx) => raw_constant_indices.push(*idx),
                GepIndex::Value(value) => {
                    dynamic_indices.push(*value);
                    raw_constant_indices.push(i32::MIN); // marker for dynamic index
                }
            }
        }

        let mut op = ods::llvm::getelementptr(
            context,
            pointer(context, 0),
            pointer,
            &dynamic_indices,
            DenseI32ArrayAttribute::new(context, &raw_constant_indices),
            TypeAttribute::new(element_type),
            location,
        );
        op.set_inbounds(Attribute::unit(context));

        self.append_op_result(op.into())
    }
}
