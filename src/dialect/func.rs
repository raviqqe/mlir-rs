use crate::ir::{operation::Builder, Location, Operation, Type, Value};

pub fn func<'c>(
    arguments: &[Value],
    results: &[Type<'c>],
    location: Location<'c>,
) -> Operation<'c> {
    Builder::new("func.func", location)
        .add_operands(&arguments)
        .add_results(&results)
        .build()
}
