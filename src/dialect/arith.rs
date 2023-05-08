use crate::ir::{operation::Builder, Location, Operation, Value};

pub fn addi<'c>(lhs: Value, rhs: Value, location: Location<'c>) -> Operation<'c> {
    binary_operator("arith.addi", lhs, rhs, location)
}

pub fn subi<'c>(lhs: Value, rhs: Value, location: Location<'c>) -> Operation<'c> {
    binary_operator("arith.subi", lhs, rhs, location)
}

pub fn muli<'c>(lhs: Value, rhs: Value, location: Location<'c>) -> Operation<'c> {
    binary_operator("arith.muli", lhs, rhs, location)
}

pub fn divi<'c>(lhs: Value, rhs: Value, location: Location<'c>) -> Operation<'c> {
    binary_operator("arith.divi", lhs, rhs, location)
}

pub fn addf<'c>(lhs: Value, rhs: Value, location: Location<'c>) -> Operation<'c> {
    binary_operator("arith.addf", lhs, rhs, location)
}

pub fn subf<'c>(lhs: Value, rhs: Value, location: Location<'c>) -> Operation<'c> {
    binary_operator("arith.subf", lhs, rhs, location)
}

pub fn mulf<'c>(lhs: Value, rhs: Value, location: Location<'c>) -> Operation<'c> {
    binary_operator("arith.mulf", lhs, rhs, location)
}

pub fn divf<'c>(lhs: Value, rhs: Value, location: Location<'c>) -> Operation<'c> {
    binary_operator("arith.divf", lhs, rhs, location)
}

fn binary_operator<'c>(
    name: &str,
    lhs: Value,
    rhs: Value,
    location: Location<'c>,
) -> Operation<'c> {
    Builder::new(name, location)
        .add_operands(&[lhs, rhs])
        .enable_result_type_inference()
        .build()
}
