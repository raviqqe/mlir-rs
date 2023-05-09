//! `scf` dialect.

use crate::ir::{operation::Builder, Location, Operation};

/// Creates a `scf.yield` operation.
pub fn r#yield(location: Location) -> Operation {
    Builder::new("scf.yield", location).build()
}
