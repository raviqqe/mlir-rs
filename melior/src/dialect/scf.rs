//! `scf` dialect.

use crate::ir::{operation::Builder, Location, Operation, Region, Value};

/// Creates a `scf.for` operation.
pub fn r#for<'c>(
    start: Value<'c>,
    end: Value<'c>,
    step: Value<'c>,
    region: Region,
    location: Location<'c>,
) -> Operation<'c> {
    Builder::new("scf.for", location)
        .add_operands(&[start, end, step])
        .add_regions(vec![region])
        .build()
}

/// Creates a `scf.yield` operation.
pub fn r#yield<'c>(values: &[Value<'c>], location: Location<'c>) -> Operation<'c> {
    Builder::new("scf.yield", location)
        .add_operands(values)
        .build()
}
