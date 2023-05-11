use crate::{dialect::DialectRegistry, utility::register_all_dialects, Context};

pub fn load_all_dialects(context: &Context) {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
}

pub fn create_test_context() -> Context {
    let context = Context::new();

    context.attach_diagnostic_handler(|d| {
        eprintln!("{}", d);
        true
    });

    load_all_dialects(&context);

    context
}
