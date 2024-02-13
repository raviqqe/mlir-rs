use super::error::{Error, OdsError};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use syn::Type;
use tblgen::{
    error::{TableGenError, WithLocation},
    record::Record,
};

macro_rules! prefixed_string {
    ($prefix:literal, $name:ident) => {
        concat!($prefix, stringify!($name))
    };
}

macro_rules! mlir_attribute {
    ($name:ident) => {
        prefixed_string!("::mlir::", $name)
    };
}

macro_rules! melior_attribute {
    ($name:ident) => {
        prefixed_string!("::melior::ir::attribute::", $name)
    };
}

static ATTRIBUTE_TYPES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();

    macro_rules! initialize_attributes {
        ($($mlir:ident => $melior:ident),* $(,)*) => {
            $(
                map.insert(
                    mlir_attribute!($mlir),
                    melior_attribute!($melior),
                );
            )*
        };
    }

    initialize_attributes!(
        ArrayAttr => ArrayAttribute,
        Attribute => Attribute,
        DenseElementsAttr => DenseElementsAttribute,
        DenseI32ArrayAttr => DenseI32ArrayAttribute,
        FlatSymbolRefAttr => FlatSymbolRefAttribute,
        FloatAttr => FloatAttribute,
        IntegerAttr => IntegerAttribute,
        StringAttr => StringAttribute,
        TypeAttr => TypeAttribute,
    );

    map
});

#[derive(Debug, Clone, Copy)]
pub struct RegionConstraint<'a>(Record<'a>);

impl<'a> RegionConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    pub fn is_variadic(&self) -> bool {
        self.0.subclass_of("VariadicRegion")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SuccessorConstraint<'a>(Record<'a>);

impl<'a> SuccessorConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    pub fn is_variadic(&self) -> bool {
        self.0.subclass_of("VariadicSuccessor")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TypeConstraint<'a>(Record<'a>);

impl<'a> TypeConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    pub fn is_optional(&self) -> bool {
        self.0.subclass_of("Optional")
    }

    pub fn is_variadic(&self) -> bool {
        self.0.subclass_of("Variadic")
    }

    // TODO Support variadic-of-variadic.
    #[allow(unused)]
    pub fn is_variadic_of_variadic(&self) -> bool {
        self.0.subclass_of("VariadicOfVariadic")
    }

    pub fn has_unfixed(&self) -> bool {
        self.is_variadic() || self.is_optional()
    }
}

#[derive(Debug, Clone)]
pub struct AttributeConstraint<'a> {
    record: Record<'a>,
    name: &'a str,
    storage_type_str: String,
    storage_type: Type,
    optional: bool,
    default: bool,
}

impl<'a> AttributeConstraint<'a> {
    pub fn new(record: Record<'a>) -> Result<Self, Error> {
        let storage_type_str = record.string_value("storageType")?;

        Ok(Self {
            name: record.name()?,
            storage_type: syn::parse_str(
                &ATTRIBUTE_TYPES
                    .get(storage_type_str.trim())
                    .copied()
                    .unwrap_or(melior_attribute!(Attribute)),
            )?,
            storage_type_str,
            optional: record.bit_value("isOptional")?,
            default: match record.string_value("defaultValue") {
                Ok(value) => !value.is_empty(),
                Err(error) => {
                    // `defaultValue` can be uninitialized.
                    if !matches!(error.error(), TableGenError::InitConversion { .. }) {
                        return Err(error.into());
                    }

                    false
                }
            },
            record,
        })
    }

    #[allow(unused)]
    pub fn is_derived(&self) -> bool {
        self.record.subclass_of("DerivedAttr")
    }

    #[allow(unused)]
    pub fn is_type(&self) -> bool {
        self.record.subclass_of("TypeAttrBase")
    }

    #[allow(unused)]
    pub fn is_symbol_ref(&self) -> bool {
        self.name == "SymbolRefAttr"
            || self.name == "FlatSymbolRefAttr"
            || self.record.subclass_of("SymbolRefAttr")
            || self.record.subclass_of("FlatSymbolRefAttr")
    }

    #[allow(unused)]
    pub fn is_enum(&self) -> bool {
        self.record.subclass_of("EnumAttrInfo")
    }

    pub fn is_optional(&self) -> bool {
        self.optional
    }

    pub fn storage_type(&self) -> &Type {
        &self.storage_type
    }

    pub fn is_unit(&self) -> bool {
        self.storage_type_str == mlir_attribute!(UnitAttr)
    }

    pub fn has_default_value(&self) -> bool {
        self.default
    }
}

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
            | TraitKind::Interface { name } => Some(&name),
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
