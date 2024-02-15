use super::error::{Error, OdsError};
use tblgen::{error::WithLocation, record::Record};

#[derive(Debug, Clone)]
pub enum Trait {
    Interface {
        name: String,
    },
    Internal {
        name: String,
    },
    Native {
        name: String,
        #[allow(unused)]
        structural: bool,
    },
    Predicate,
}

impl Trait {
    pub fn new(definition: Record) -> Result<Self, Error> {
        Ok(if definition.subclass_of("PredTrait") {
            Self::Predicate
        } else if definition.subclass_of("InterfaceTrait") {
            Self::Interface {
                name: Self::build_name(definition)?,
            }
        } else if definition.subclass_of("NativeTrait") {
            Self::Native {
                name: Self::build_name(definition)?,
                structural: definition.subclass_of("StructuralOpTrait"),
            }
        } else if definition.subclass_of("GenInternalTrait") {
            Self::Internal {
                name: definition.string_value("trait")?,
            }
        } else {
            return Err(OdsError::InvalidTrait.with_location(definition).into());
        })
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Native { name, .. } | Self::Internal { name } | Self::Interface { name } => {
                Some(name)
            }
            Self::Predicate => None,
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
