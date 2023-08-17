use once_cell::sync::Lazy;
use std::{collections::HashMap, ops::Deref};
use tblgen::record::Record;

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

pub static ATTRIBUTE_TYPES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
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

#[allow(unused)]
impl<'a> RegionConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    pub fn is_variadic(&self) -> bool {
        self.0.subclass_of("VariadicRegion")
    }
}

impl<'a> Deref for RegionConstraint<'a> {
    type Target = Record<'a>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SuccessorConstraint<'a>(Record<'a>);

#[allow(unused)]
impl<'a> SuccessorConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    pub fn is_variadic(&self) -> bool {
        self.0.subclass_of("VariadicSuccessor")
    }
}

impl<'a> Deref for SuccessorConstraint<'a> {
    type Target = Record<'a>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TypeConstraint<'a>(Record<'a>);

#[allow(unused)]
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

    pub fn is_variadic_of_variadic(&self) -> bool {
        self.0.subclass_of("VariadicOfVariadic")
    }

    pub fn is_variable_length(&self) -> bool {
        self.is_variadic() || self.is_optional()
    }
}

impl<'a> Deref for TypeConstraint<'a> {
    type Target = Record<'a>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttributeConstraint<'a>(Record<'a>);

#[allow(unused)]
impl<'a> AttributeConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    pub fn is_derived(&self) -> bool {
        self.0.subclass_of("DerivedAttr")
    }

    pub fn is_type_attr(&self) -> bool {
        self.0.subclass_of("TypeAttrBase")
    }

    pub fn is_symbol_ref_attr(&self) -> bool {
        self.0.name() == Ok("SymbolRefAttr")
            || self.0.name() == Ok("FlatSymbolRefAttr")
            || self.0.subclass_of("SymbolRefAttr")
            || self.0.subclass_of("FlatSymbolRefAttr")
    }

    pub fn is_enum_attr(&self) -> bool {
        self.0.subclass_of("EnumAttrInfo")
    }

    pub fn is_optional(&self) -> bool {
        self.0.bit_value("isOptional").unwrap_or(false)
    }

    pub fn storage_type(&self) -> &'static str {
        self.0
            .string_value("storageType")
            .ok()
            .and_then(|v| ATTRIBUTE_TYPES.get(v.as_str().trim()))
            .copied()
            .unwrap_or(melior_attribute!(Attribute))
    }

    pub fn is_unit(&self) -> bool {
        self.0
            .string_value("storageType")
            .map(|v| v == mlir_attribute!(UnitAttr))
            .unwrap_or(false)
    }

    pub fn has_default_value(&self) -> bool {
        self.0
            .string_value("defaultValue")
            .map(|s| !s.is_empty())
            .unwrap_or(false)
    }
}

impl<'a> Deref for AttributeConstraint<'a> {
    type Target = Record<'a>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub enum TraitKind {
    Native { name: String, is_structural: bool },
    Pred {},
    Internal { name: String },
    Interface { name: String },
}

#[derive(Debug, Clone)]
pub struct Trait<'a> {
    kind: TraitKind,
    def: Record<'a>,
}

#[allow(unused)]
impl<'a> Trait<'a> {
    pub fn new(def: Record<'a>) -> Self {
        Self {
            def,
            kind: if def.subclass_of("PredTrait") {
                TraitKind::Pred {}
            } else if def.subclass_of("InterfaceTrait") {
                TraitKind::Interface {
                    name: Self::name(def),
                }
            } else if def.subclass_of("NativeTrait") {
                TraitKind::Native {
                    name: Self::name(def),
                    is_structural: def.subclass_of("StructuralOpTrait"),
                }
            } else if def.subclass_of("GenInternalTrait") {
                TraitKind::Internal {
                    name: def
                        .string_value("trait")
                        .expect("trait def has trait value"),
                }
            } else {
                unreachable!("invalid trait")
            },
        }
    }

    pub fn has_name(&self, expected_name: &str) -> bool {
        match &self.kind {
            TraitKind::Native { name, .. }
            | TraitKind::Internal { name }
            | TraitKind::Interface { name } => expected_name == name,
            TraitKind::Pred {} => false,
        }
    }

    fn name(def: Record) -> String {
        let r#trait = def
            .string_value("trait")
            .expect("trait def has trait value");

        if let Some(namespace) = def
            .string_value("cppNamespace")
            .ok()
            .filter(|namespace| !namespace.is_empty())
        {
            format!("{}::{}", namespace, r#trait)
        } else {
            r#trait
        }
    }

    pub fn kind(&self) -> &TraitKind {
        &self.kind
    }
}

impl<'a> Deref for Trait<'a> {
    type Target = Record<'a>;

    fn deref(&self) -> &Self::Target {
        &self.def
    }
}
