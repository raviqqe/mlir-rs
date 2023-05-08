use crate::{dialect::Registry, utility::register_all_dialects, Context};

pub fn load_all_dialects(context: &Context) {
    let registry = Registry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
}
