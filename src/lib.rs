pub mod attribute;
pub mod block;
pub mod context;
pub mod dialect;
pub mod dialect_registry;
pub mod identifier;
pub mod location;
pub mod module;
pub mod operation;
pub mod operation_state;
pub mod region;
pub mod string_ref;
pub mod r#type;
pub mod utility;
pub mod value;

#[cfg(test)]
mod tests {
    use crate::{
        context::Context, dialect_registry::DialectRegistry, location::Location, module::Module,
    };

    #[test]
    fn build_module() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));

        assert_eq!(module.as_operation().print().as_str().to_str().unwrap(), "");
    }

    #[test]
    fn build_module_with_dialect() {
        let registry = DialectRegistry::new();
        let context = Context::new();
        context.append_dialect_registry(&registry);
        let module = Module::new(Location::unknown(&context));

        assert_eq!(module.as_operation().print().as_str().to_str().unwrap(), "");
    }
}
