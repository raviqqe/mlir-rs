use melior::{dialect::DialectRegistry, utility::register_all_dialects, Context};

// Generate rust code for the toy dialect.
// This will expand to a module
melior::dialect! {
    name: "toy",
    td_file: "examples/toy/src/include/toy.td",
}

fn main() {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context.set_allow_unregistered_dialects(true);
}
