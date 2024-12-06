//! Trait that extends the melior Block type to aid in codegen and consistency.

use core::fmt;

use crate::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm::r#type::pointer,
        ods,
    },
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute,
        },
        r#type::IntegerType,
        Attribute, Block, Location, Operation, Type, Value, ValueLike,
    },
    Context, Error,
};

/// Index types for LLVM GEP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GepIndex<'c, 'a> {
    /// A compile time known index.
    Const(i32),
    /// A runtime value index.
    Value(Value<'c, 'a>),
}

pub trait BlockExt<'ctx> {
    fn arg(&self, idx: usize) -> Result<Value<'ctx, '_>, Error>;

    /// Creates an arith.cmpi operation.
    fn cmpi(
        &self,
        context: &'ctx Context,
        pred: CmpiPredicate,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn extui(
        &self,
        lhs: Value<'ctx, '_>,
        target_type: Type<'ctx>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn extsi(
        &self,
        lhs: Value<'ctx, '_>,
        target_type: Type<'ctx>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn trunci(
        &self,
        lhs: Value<'ctx, '_>,
        target_type: Type<'ctx>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn shrui(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn shli(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn addi(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn subi(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn divui(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn divsi(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn xori(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn ori(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn andi(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    fn muli(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Appends the operation and returns the first result.
    fn append_op_result(&self, operation: Operation<'ctx>) -> Result<Value<'ctx, '_>, Error>;

    /// Creates a constant of the given integer bit width. Do not use for felt252.
    fn const_int<T: fmt::Display>(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        value: T,
        bits: u32,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Creates a constant of the given integer type. Do not use for felt252.
    fn const_int_from_type<T: fmt::Display>(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        value: T,
        int_type: Type<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Uses a llvm::extract_value operation to return the value at the given index of a container (e.g struct).
    fn extract_value(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        container: Value<'ctx, '_>,
        value_type: Type<'ctx>,
        index: usize,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Uses a llvm::insert_value operation to insert the value at the given index of a container (e.g struct),
    /// the result is the container with the value.
    fn insert_value(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        container: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
        index: usize,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Uses a llvm::insert_value operation to insert the values starting from index 0 into a container (e.g struct),
    /// the result is the container with the values.
    fn insert_values<'block>(
        &'block self,
        context: &'ctx Context,
        location: Location<'ctx>,
        container: Value<'ctx, 'block>,
        values: &[Value<'ctx, 'block>],
    ) -> Result<Value<'ctx, 'block>, Error>;

    /// Loads a value from the given addr.
    fn load(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        addr: Value<'ctx, '_>,
        value_type: Type<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Allocates the given number of elements of type in memory on the stack, returning a opaque pointer.
    fn alloca(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        element_type: Type<'ctx>,
        element_count: Value<'ctx, '_>,
        align: usize,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Allocates one element of the given type in memory on the stack, returning a opaque pointer.
    fn alloca1(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        element_type: Type<'ctx>,
        align: usize,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Allocates one integer of the given bit width.
    fn alloca_int(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        bits: u32,
        align: usize,
    ) -> Result<Value<'ctx, '_>, Error>;

    /// Stores a value at the given addr.
    fn store(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        addr: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
    ) -> Result<(), Error>;

    /// Creates a memcpy operation.
    fn memcpy(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        src: Value<'ctx, '_>,
        dst: Value<'ctx, '_>,
        len_bytes: Value<'ctx, '_>,
    );

    /// Creates a getelementptr operation. Returns a pointer to the indexed element.
    /// This method allows combining both compile time indexes and runtime value indexes.
    ///
    /// See:
    /// - https://llvm.org/docs/LangRef.html#getelementptr-instruction
    /// - https://llvm.org/docs/GetElementPtr.html
    ///
    /// Get Element Pointer is used to index into pointers, it uses the given
    /// element type to compute the offsets, it allows indexing deep into a structure (field of field of a ptr for example),
    /// this is why it accepts a array of indexes, it indexes through the list, offsetting depending on the element type,
    /// for example it knows when you index into a struct field, the following index will use the struct field type for offsets, etc.
    ///
    /// Address computation is done at compile time.
    ///
    /// Note: This GEP sets the inbounds attribute:
    ///
    /// The base pointer has an in bounds address of the allocated object that it is based on. This means that it points into that allocated object, or to its end. Note that the object does not have to be live anymore; being in-bounds of a deallocated object is sufficient.
    ///
    /// During the successive addition of offsets to the address, the resulting pointer must remain in bounds of the allocated object at each step.
    fn gep(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        ptr: Value<'ctx, '_>,
        indexes: &[GepIndex<'ctx, '_>],
        element_type: Type<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error>;
}

impl<'ctx> BlockExt<'ctx> for Block<'ctx> {
    #[inline]
    fn arg(&self, idx: usize) -> Result<Value<'ctx, '_>, Error> {
        Ok(self.argument(idx)?.into())
    }

    #[inline]
    fn cmpi(
        &self,
        context: &'ctx Context,
        pred: CmpiPredicate,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::cmpi(context, pred, lhs, rhs, location))
    }

    #[inline]
    fn extsi(
        &self,
        lhs: Value<'ctx, '_>,
        target_type: Type<'ctx>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::extsi(lhs, target_type, location))
    }

    #[inline]
    fn extui(
        &self,
        lhs: Value<'ctx, '_>,
        target_type: Type<'ctx>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::extui(lhs, target_type, location))
    }

    #[inline]
    fn trunci(
        &self,
        lhs: Value<'ctx, '_>,
        target_type: Type<'ctx>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::trunci(lhs, target_type, location))
    }

    #[inline]
    fn shli(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::shli(lhs, rhs, location))
    }

    #[inline]
    fn shrui(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::shrui(lhs, rhs, location))
    }

    #[inline]
    fn addi(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::addi(lhs, rhs, location))
    }

    #[inline]
    fn subi(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::subi(lhs, rhs, location))
    }

    #[inline]
    fn divui(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::divui(lhs, rhs, location))
    }

    #[inline]
    fn divsi(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::divsi(lhs, rhs, location))
    }

    #[inline]
    fn xori(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::xori(lhs, rhs, location))
    }

    #[inline]
    fn ori(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::ori(lhs, rhs, location))
    }

    #[inline]
    fn andi(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::andi(lhs, rhs, location))
    }

    #[inline]
    fn muli(
        &self,
        lhs: Value<'ctx, '_>,
        rhs: Value<'ctx, '_>,
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(arith::muli(lhs, rhs, location))
    }

    #[inline]
    fn const_int<T: fmt::Display>(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        value: T,
        bits: u32,
    ) -> Result<Value<'ctx, '_>, Error> {
        let ty = IntegerType::new(context, bits).into();
        self.append_op_result(
            ods::arith::constant(
                context,
                ty,
                Attribute::parse(context, &format!("{} : {}", value, ty)).unwrap(),
                location,
            )
            .into(),
        )
    }

    #[inline]
    fn const_int_from_type<T: fmt::Display>(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        value: T,
        ty: Type<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(
            ods::arith::constant(
                context,
                ty,
                Attribute::parse(context, &format!("{} : {}", value, ty)).unwrap(),
                location,
            )
            .into(),
        )
    }

    #[inline]
    fn extract_value(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        container: Value<'ctx, '_>,
        value_type: Type<'ctx>,
        index: usize,
    ) -> Result<Value<'ctx, '_>, Error> {
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
        context: &'ctx Context,
        location: Location<'ctx>,
        container: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
        index: usize,
    ) -> Result<Value<'ctx, '_>, Error> {
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
        context: &'ctx Context,
        location: Location<'ctx>,
        mut container: Value<'ctx, 'block>,
        values: &[Value<'ctx, 'block>],
    ) -> Result<Value<'ctx, 'block>, Error> {
        for (i, value) in values.iter().enumerate() {
            container = self.insert_value(context, location, container, *value, i)?;
        }
        Ok(container)
    }

    #[inline]
    fn store(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        addr: Value<'ctx, '_>,
        value: Value<'ctx, '_>,
    ) -> Result<(), Error> {
        self.append_operation(ods::llvm::store(context, value, addr, location).into());
        Ok(())
    }

    // Use this only when returning the result. Otherwise, append_operation is fine.
    #[inline]
    fn append_op_result(&self, operation: Operation<'ctx>) -> Result<Value<'ctx, '_>, Error> {
        Ok(self.append_operation(operation).result(0)?.into())
    }

    #[inline]
    fn load(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        addr: Value<'ctx, '_>,
        value_type: Type<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
        self.append_op_result(ods::llvm::load(context, value_type, addr, location).into())
    }

    #[inline]
    fn memcpy(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        src: Value<'ctx, '_>,
        dst: Value<'ctx, '_>,
        len_bytes: Value<'ctx, '_>,
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
        context: &'ctx Context,
        location: Location<'ctx>,
        element_type: Type<'ctx>,
        element_count: Value<'ctx, '_>,
        align: usize,
    ) -> Result<Value<'ctx, '_>, Error> {
        let mut op = ods::llvm::alloca(
            context,
            pointer(context, 0),
            element_count,
            TypeAttribute::new(element_type),
            location,
        );

        op.set_element_type(TypeAttribute::new(element_type));
        op.set_alignment(IntegerAttribute::new(
            IntegerType::new(context, 64).into(),
            align.try_into().unwrap(),
        ));

        self.append_op_result(op.into())
    }

    #[inline]
    fn alloca1(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        element_type: Type<'ctx>,
        align: usize,
    ) -> Result<Value<'ctx, '_>, Error> {
        let element_count = self.const_int(context, location, 1, 64)?;
        self.alloca(context, location, element_type, element_count, align)
    }

    #[inline]
    fn alloca_int(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        bits: u32,
        align: usize,
    ) -> Result<Value<'ctx, '_>, Error> {
        let element_count = self.const_int(context, location, 1, 64)?;
        self.alloca(
            context,
            location,
            IntegerType::new(context, bits).into(),
            element_count,
            align,
        )
    }

    #[inline]
    fn gep(
        &self,
        context: &'ctx Context,
        location: Location<'ctx>,
        ptr: Value<'ctx, '_>,
        indexes: &[GepIndex<'ctx, '_>],
        element_type: Type<'ctx>,
    ) -> Result<Value<'ctx, '_>, Error> {
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
            ptr,
            &dynamic_indices,
            DenseI32ArrayAttribute::new(context, &raw_constant_indices),
            TypeAttribute::new(element_type),
            location,
        );
        op.set_inbounds(Attribute::unit(context));

        self.append_op_result(op.into())
    }
}
