use super::error::{Error, OdsError};
use tblgen::{error::WithLocation, record::Record};

#[derive(Debug, Clone)]
enum TraitKind {
    Native {
        name: String,
        #[allow(unused)]
        structural: bool,
    },
    Predicate,
    Internal {
        name: String,
    },
    Interface {
        name: String,
    },
}

#[derive(Debug, Clone)]
pub struct Trait {
    kind: TraitKind,
}

impl Trait {
    pub fn new(definition: Record) -> Result<Self, Error> {
        Ok(Self {
            kind: if definition.subclass_of("PredTrait") {
                TraitKind::Predicate
            } else if definition.subclass_of("InterfaceTrait") {
                TraitKind::Interface {
                    name: Self::build_name(definition)?,
                }
            } else if definition.subclass_of("NativeTrait") {
                TraitKind::Native {
                    name: Self::build_name(definition)?,
                    structural: definition.subclass_of("StructuralOpTrait"),
                }
            } else if definition.subclass_of("GenInternalTrait") {
                TraitKind::Internal {
                    name: definition.string_value("trait")?,
                }
            } else {
                return Err(OdsError::InvalidTrait.with_location(definition).into());
            },
        })
    }

    pub fn name(&self) -> Option<&str> {
        match &self.kind {
            TraitKind::Native { name, .. }
            | TraitKind::Internal { name }
            | TraitKind::Interface { name } => Some(name),
            TraitKind::Predicate => None,
        }
    }

    fn build_name(definition: Record) -> Result<String, Error> {
        let r#trait = definition.string_value("trait")?;
        let namespace = definition.string_value("cppNamespace")?;

        Ok(if namespace.is_empty() {
            r#trait
        } else {
            format!("{namespace}::{trait}")
        })
    }
}
