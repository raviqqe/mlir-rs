use super::builtin::BuiltinBlockExt;
use crate::{
    dialect::{
        arith::{
            addi, andi, cmpi, divsi, divui, extsi, extui, muli, ori, shli, shrui, subi, trunci,
            xori, CmpiPredicate,
        },
        ods,
    },
    ir::{r#type::IntegerType, Attribute, Block, Location, Type, Value},
    Context, Error,
};
use core::fmt;

pub trait ArithBlockExt<'c>: BuiltinBlockExt<'c> {
    /// Creates an `arith.cmpi` operation.
    fn cmpi(
        &self,
        context: &'c Context,
        pred: CmpiPredicate,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates an `arith.extui` operation.
    fn extui(
        &self,
        lhs: Value<'c, '_>,
        target_type: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    fn extsi(
        &self,
        lhs: Value<'c, '_>,
        target_type: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    fn trunci(
        &self,
        lhs: Value<'c, '_>,
        target_type: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    fn shrui(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    fn shli(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    fn addi(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    fn subi(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    fn divui(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    fn divsi(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    fn xori(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    fn ori(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    fn andi(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    fn muli(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates a constant of the given integer bit width. Do not use for felt252.
    fn const_int<T: fmt::Display>(
        &self,
        context: &'c Context,
        location: Location<'c>,
        value: T,
        bits: u32,
    ) -> Result<Value<'c, '_>, Error>;

    /// Creates a constant of the given integer type. Do not use for felt252.
    fn const_int_from_type<T: fmt::Display>(
        &self,
        context: &'c Context,
        location: Location<'c>,
        value: T,
        int_type: Type<'c>,
    ) -> Result<Value<'c, '_>, Error>;
}

impl<'c> ArithBlockExt<'c> for Block<'c> {
    #[inline]
    fn cmpi(
        &self,
        context: &'c Context,
        pred: CmpiPredicate,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(cmpi(context, pred, lhs, rhs, location))
    }

    #[inline]
    fn extsi(
        &self,
        lhs: Value<'c, '_>,
        target_type: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(extsi(lhs, target_type, location))
    }

    #[inline]
    fn extui(
        &self,
        lhs: Value<'c, '_>,
        target_type: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(extui(lhs, target_type, location))
    }

    #[inline]
    fn trunci(
        &self,
        lhs: Value<'c, '_>,
        target_type: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(trunci(lhs, target_type, location))
    }

    #[inline]
    fn shli(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(shli(lhs, rhs, location))
    }

    #[inline]
    fn shrui(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(shrui(lhs, rhs, location))
    }

    #[inline]
    fn addi(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(addi(lhs, rhs, location))
    }

    #[inline]
    fn subi(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(subi(lhs, rhs, location))
    }

    #[inline]
    fn divui(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(divui(lhs, rhs, location))
    }

    #[inline]
    fn divsi(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(divsi(lhs, rhs, location))
    }

    #[inline]
    fn xori(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(xori(lhs, rhs, location))
    }

    #[inline]
    fn ori(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(ori(lhs, rhs, location))
    }

    #[inline]
    fn andi(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(andi(lhs, rhs, location))
    }

    #[inline]
    fn muli(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(muli(lhs, rhs, location))
    }

    #[inline]
    fn const_int<T: fmt::Display>(
        &self,
        context: &'c Context,
        location: Location<'c>,
        value: T,
        bits: u32,
    ) -> Result<Value<'c, '_>, Error> {
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
        context: &'c Context,
        location: Location<'c>,
        value: T,
        r#type: Type<'c>,
    ) -> Result<Value<'c, '_>, Error> {
        self.append_op_result(
            ods::arith::constant(
                context,
                r#type,
                Attribute::parse(context, &format!("{value} : {type}")).unwrap(),
                location,
            )
            .into(),
        )
    }
}
