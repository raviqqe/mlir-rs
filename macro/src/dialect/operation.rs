mod accessors;
mod builder;

use self::builder::OperationBuilder;
use super::utility::{sanitize_documentation, sanitize_snake_case_name};
use crate::dialect::{
    error::{Error, ExpectedSuperClassError},
    types::{AttributeConstraint, RegionConstraint, SuccessorConstraint, Trait, TypeConstraint},
};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote, ToTokens, TokenStreamExt};
use syn::{parse_quote, Type};
use tblgen::{error::WithLocation, record::Record};

#[derive(Debug, Clone, Copy)]
pub enum ElementKind {
    Operand,
    Result,
}

impl ElementKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Operand => "operand",
            Self::Result => "result",
        }
    }
}

#[derive(Debug, Clone)]
pub enum FieldKind<'a> {
    Element {
        kind: ElementKind,
        constraint: TypeConstraint<'a>,
        sequence_info: SequenceInfo,
        variadic_kind: VariadicKind,
    },
    Attribute {
        constraint: AttributeConstraint<'a>,
    },
    Successor {
        constraint: SuccessorConstraint<'a>,
        sequence_info: SequenceInfo,
    },
    Region {
        constraint: RegionConstraint<'a>,
        sequence_info: SequenceInfo,
    },
}

impl<'a> FieldKind<'a> {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Element { kind, .. } => kind.as_str(),
            Self::Attribute { .. } => "attribute",
            Self::Successor { .. } => "successor",
            Self::Region { .. } => "region",
        }
    }

    pub fn is_optional(&self) -> bool {
        match self {
            Self::Element { constraint, .. } => constraint.is_optional(),
            Self::Attribute { constraint, .. } => {
                constraint.is_optional() || constraint.has_default_value()
            }
            Self::Successor { .. } | Self::Region { .. } => false,
        }
    }

    pub fn is_result(&self) -> bool {
        matches!(
            self,
            Self::Element {
                kind: ElementKind::Result,
                ..
            }
        )
    }

    pub fn param_type(&self) -> Type {
        match self {
            Self::Element {
                kind, constraint, ..
            } => {
                let base_type: Type = match kind {
                    ElementKind::Operand => {
                        parse_quote!(::melior::ir::Value<'c, '_>)
                    }
                    ElementKind::Result => {
                        parse_quote!(::melior::ir::Type<'c>)
                    }
                };
                if constraint.is_variadic() {
                    parse_quote! { &[#base_type] }
                } else {
                    base_type
                }
            }
            Self::Attribute { constraint } => {
                if constraint.is_unit() {
                    parse_quote!(bool)
                } else {
                    let r#type: Type = syn::parse_str(constraint.storage_type())
                        .expect("storage type strings are valid");
                    parse_quote!(#r#type<'c>)
                }
            }
            Self::Successor { constraint, .. } => {
                let r#type: Type = parse_quote!(&::melior::ir::Block<'c>);
                if constraint.is_variadic() {
                    parse_quote!(&[#r#type])
                } else {
                    r#type
                }
            }
            Self::Region { constraint, .. } => {
                let r#type: Type = parse_quote!(::melior::ir::Region<'c>);
                if constraint.is_variadic() {
                    parse_quote!(Vec<#r#type>)
                } else {
                    r#type
                }
            }
        }
    }

    fn create_result_type(r#type: Type) -> Type {
        parse_quote!(Result<#r#type, ::melior::Error>)
    }

    fn create_iterator_type(r#type: Type) -> Type {
        parse_quote!(impl Iterator<Item = #r#type>)
    }

    pub fn return_type(&self) -> Type {
        match self {
            Self::Element {
                kind,
                constraint,
                variadic_kind,
                ..
            } => {
                let base_type: Type = match kind {
                    ElementKind::Operand => {
                        parse_quote!(::melior::ir::Value<'c, '_>)
                    }
                    ElementKind::Result => {
                        parse_quote!(::melior::ir::operation::OperationResult<'c, '_>)
                    }
                };
                if !constraint.is_variadic() {
                    Self::create_result_type(base_type)
                } else if let VariadicKind::AttrSized {} = variadic_kind {
                    Self::create_result_type(Self::create_iterator_type(base_type))
                } else {
                    Self::create_iterator_type(base_type)
                }
            }
            Self::Attribute { constraint } => {
                if constraint.is_unit() {
                    parse_quote!(bool)
                } else {
                    Self::create_result_type(self.param_type())
                }
            }
            Self::Successor { constraint, .. } => {
                let r#type: Type = parse_quote!(::melior::ir::BlockRef<'c, '_>);
                if constraint.is_variadic() {
                    Self::create_iterator_type(r#type)
                } else {
                    Self::create_result_type(r#type)
                }
            }
            Self::Region { constraint, .. } => {
                let r#type: Type = parse_quote!(::melior::ir::RegionRef<'c, '_>);
                if constraint.is_variadic() {
                    Self::create_iterator_type(r#type)
                } else {
                    Self::create_result_type(r#type)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SequenceInfo {
    index: usize,
    len: usize,
}

#[derive(Clone, Debug)]
pub enum VariadicKind {
    Simple {
        seen_variable_length: bool,
    },
    SameSize {
        num_variable_length: usize,
        num_preceding_simple: usize,
        num_preceding_variadic: usize,
    },
    AttrSized {},
}

#[derive(Debug, Clone)]
pub struct OperationField<'a> {
    pub(crate) name: &'a str,
    pub(crate) sanitized_name: Ident,
    pub(crate) kind: FieldKind<'a>,
}

impl<'a> OperationField<'a> {
    pub fn new(name: &'a str, kind: FieldKind<'a>) -> Self {
        Self {
            name,
            sanitized_name: sanitize_snake_case_name(name),
            kind,
        }
    }

    pub fn new_attribute(name: &'a str, constraint: AttributeConstraint<'a>) -> Self {
        Self::new(name, FieldKind::Attribute { constraint })
    }

    pub fn new_region(
        name: &'a str,
        constraint: RegionConstraint<'a>,
        sequence_info: SequenceInfo,
    ) -> Self {
        Self::new(
            name,
            FieldKind::Region {
                constraint,
                sequence_info,
            },
        )
    }

    pub fn new_successor(
        name: &'a str,
        constraint: SuccessorConstraint<'a>,
        sequence_info: SequenceInfo,
    ) -> Self {
        Self::new(
            name,
            FieldKind::Successor {
                constraint,
                sequence_info,
            },
        )
    }

    pub fn new_operand(
        name: &'a str,
        tc: TypeConstraint<'a>,
        seq_info: SequenceInfo,
        variadic_kind: VariadicKind,
    ) -> Self {
        Self::new_element(name, tc, ElementKind::Operand, seq_info, variadic_kind)
    }

    pub fn new_result(
        name: &'a str,
        tc: TypeConstraint<'a>,
        seq_info: SequenceInfo,
        variadic_kind: VariadicKind,
    ) -> Self {
        Self::new_element(name, tc, ElementKind::Result, seq_info, variadic_kind)
    }

    fn new_element(
        name: &'a str,
        constraint: TypeConstraint<'a>,
        kind: ElementKind,
        sequence_info: SequenceInfo,
        variadic_kind: VariadicKind,
    ) -> Self {
        Self::new(
            name,
            FieldKind::Element {
                kind,
                constraint,
                sequence_info,
                variadic_kind,
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct Operation<'a> {
    pub(crate) dialect: Record<'a>,
    pub(crate) short_name: &'a str,
    pub(crate) full_name: String,
    pub(crate) class_name: &'a str,
    pub(crate) fields: Vec<OperationField<'a>>,
    pub(crate) can_infer_type: bool,
    pub(crate) summary: String,
    pub(crate) description: String,
}

impl<'a> Operation<'a> {
    pub fn from_def(def: Record<'a>) -> Result<Self, Error> {
        let dialect = def.def_value("opDialect")?;

        let mut work_list: Vec<_> = vec![def.list_value("traits")?];
        let mut traits = Vec::new();
        while let Some(trait_def) = work_list.pop() {
            for v in trait_def.iter() {
                let trait_def: Record = v
                    .try_into()
                    .map_err(|e: tblgen::Error| e.set_location(def))?;
                if trait_def.subclass_of("TraitList") {
                    work_list.push(trait_def.list_value("traits")?);
                } else {
                    if trait_def.subclass_of("Interface") {
                        work_list.push(trait_def.list_value("baseInterfaces")?);
                    }
                    traits.push(Trait::new(trait_def)?)
                }
            }
        }

        let successors_dag = def.dag_value("successors")?;
        let len = successors_dag.num_args();
        let successors = successors_dag.args().enumerate().map(|(i, (n, v))| {
            Result::<_, Error>::Ok(OperationField::new_successor(
                n,
                SuccessorConstraint::new(
                    v.try_into()
                        .map_err(|e: tblgen::Error| e.set_location(def))?,
                ),
                SequenceInfo { index: i, len },
            ))
        });

        let regions_dag = def.dag_value("regions").expect("operation has regions");
        let len = regions_dag.num_args();
        let regions = regions_dag.args().enumerate().map(|(i, (n, v))| {
            Ok(OperationField::new_region(
                n,
                RegionConstraint::new(
                    v.try_into()
                        .map_err(|e: tblgen::Error| e.set_location(def))?,
                ),
                SequenceInfo { index: i, len },
            ))
        });

        // Creates an initial `VariadicKind` instance based on SameSize and AttrSized
        // traits.
        let initial_variadic_kind = |num_variable_length: usize, kind_name_upper: &str| {
            let same_size_trait = format!("::mlir::OpTrait::SameVariadic{}Size", kind_name_upper);
            let attr_sized = format!("::mlir::OpTrait::AttrSized{}Segments", kind_name_upper);
            if num_variable_length <= 1 {
                VariadicKind::Simple {
                    seen_variable_length: false,
                }
            } else if traits.iter().any(|t| t.has_name(&same_size_trait)) {
                VariadicKind::SameSize {
                    num_variable_length,
                    num_preceding_simple: 0,
                    num_preceding_variadic: 0,
                }
            } else if traits.iter().any(|t| t.has_name(&attr_sized)) {
                VariadicKind::AttrSized {}
            } else {
                unimplemented!("unsupported {} structure", kind_name_upper)
            }
        };

        // Updates the given `VariadicKind` and returns the original value.
        let update_variadic_kind = |tc: &TypeConstraint, variadic_kind: &mut VariadicKind| {
            let orig_variadic_kind = variadic_kind.clone();
            match variadic_kind {
                VariadicKind::Simple {
                    seen_variable_length,
                } => {
                    if tc.is_variable_length() {
                        *seen_variable_length = true;
                    }
                    variadic_kind.clone()
                }
                VariadicKind::SameSize {
                    num_preceding_simple,
                    num_preceding_variadic,
                    ..
                } => {
                    if tc.is_variable_length() {
                        *num_preceding_variadic += 1;
                    } else {
                        *num_preceding_simple += 1;
                    }
                    orig_variadic_kind
                }
                VariadicKind::AttrSized {} => variadic_kind.clone(),
            }
        };

        let results_dag = def.dag_value("results")?;
        let results = results_dag.args().map(|(n, arg)| {
            let mut arg_def: Record = arg
                .try_into()
                .map_err(|e: tblgen::Error| e.set_location(def))?;

            if arg_def.subclass_of("OpVariable") {
                arg_def = arg_def.def_value("constraint")?;
            }

            Ok((n, TypeConstraint::new(arg_def)))
        });
        let num_results = results.clone().count();
        let num_variable_length_results = results
            .clone()
            .filter(|res| {
                res.as_ref()
                    .map(|(_, tc)| tc.is_variable_length())
                    .unwrap_or_default()
            })
            .count();
        let mut kind = initial_variadic_kind(num_variable_length_results, "Result");
        let results = results.enumerate().map(|(i, res)| {
            res.map(|(n, tc)| {
                let current_kind = update_variadic_kind(&tc, &mut kind);
                OperationField::new_result(
                    n,
                    tc,
                    SequenceInfo {
                        index: i,
                        len: num_results,
                    },
                    current_kind,
                )
            })
        });

        let arguments_dag = def.dag_value("arguments")?;
        let arguments = arguments_dag.args().map(|(name, arg)| {
            let mut arg_def: Record = arg
                .try_into()
                .map_err(|e: tblgen::Error| e.set_location(def))?;

            if arg_def.subclass_of("OpVariable") {
                arg_def = arg_def.def_value("constraint")?;
            }

            Ok((name, arg_def))
        });

        let operands = arguments.clone().filter_map(|res| {
            res.map(|(n, arg_def)| {
                if arg_def.subclass_of("TypeConstraint") {
                    Some((n, TypeConstraint::new(arg_def)))
                } else {
                    None
                }
            })
            .transpose()
        });
        let num_operands = operands.clone().count();
        let num_variable_length_operands = operands
            .clone()
            .filter(|res| {
                res.as_ref()
                    .map(|(_, tc)| tc.is_variable_length())
                    .unwrap_or_default()
            })
            .count();
        let mut kind = initial_variadic_kind(num_variable_length_operands, "Operand");
        let operands = operands.enumerate().map(|(i, res)| {
            res.map(|(name, tc)| {
                let current_kind = update_variadic_kind(&tc, &mut kind);
                OperationField::new_operand(
                    name,
                    tc,
                    SequenceInfo {
                        index: i,
                        len: num_operands,
                    },
                    current_kind,
                )
            })
        });

        let attributes = arguments.clone().filter_map(|res| {
            res.map(|(name, arg_def)| {
                if arg_def.subclass_of("Attr") {
                    assert!(!name.is_empty());
                    assert!(!arg_def.subclass_of("DerivedAttr"));
                    Some(OperationField::new_attribute(
                        name,
                        AttributeConstraint::new(arg_def),
                    ))
                } else {
                    None
                }
            })
            .transpose()
        });

        let derived_attrs = def.values().map(Ok).filter_map(|val| {
            val.and_then(|val| {
                if let Ok(def) = Record::try_from(val) {
                    if def.subclass_of("Attr") {
                        def.subclass_of("DerivedAttr")
                            .then_some(())
                            .ok_or_else(|| {
                                ExpectedSuperClassError("DerivedAttr".into()).with_location(def)
                            })?;
                        return Ok(Some(OperationField::new_attribute(
                            def.name()?,
                            AttributeConstraint::new(def),
                        )));
                    }
                }
                Ok(None)
            })
            .transpose()
        });

        let fields = successors
            .chain(regions)
            .chain(results)
            .chain(operands)
            .chain(attributes)
            .chain(derived_attrs)
            .collect::<Result<Vec<_>, _>>()?;

        let name = def.name()?;
        let class_name = if name.contains('_') && !name.starts_with('_') {
            // Trim dialect prefix from name
            name.split('_')
                .nth(1)
                .expect("string contains separator '_'")
        } else {
            name
        };

        let can_infer_type = traits.iter().any(|t| {
            (t.has_name("::mlir::OpTrait::FirstAttrDerivedResultType")
                || t.has_name("::mlir::OpTrait::SameOperandsAndResultType"))
                && num_variable_length_results == 0
                || t.has_name("::mlir::InferTypeOpInterface::Trait") && regions_dag.num_args() == 0
        });

        let short_name = def.str_value("opName")?;
        let dialect_name = dialect.string_value("name")?;
        let full_name = if !dialect_name.is_empty() {
            format!("{}.{}", dialect_name, short_name)
        } else {
            short_name.into()
        };

        let summary = def.str_value("summary").unwrap_or(short_name);
        let description = def.str_value("description").unwrap_or("");

        let summary = if !summary.is_empty() {
            format!(
                "[`{}`]({}) operation: {}",
                short_name,
                class_name,
                summary[0..1].to_uppercase() + &summary[1..]
            )
        } else {
            format!("[`{}`]({}) operation", short_name, class_name)
        };
        let description = unindent::unindent(description);

        Ok(Self {
            dialect,
            short_name,
            full_name,
            class_name,
            fields,
            can_infer_type,
            summary,
            description,
        })
    }
}

impl<'a> ToTokens for Operation<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let class_name = format_ident!("{}", &self.class_name);
        let name = &self.full_name;
        let accessors = self.fields.iter().map(|field| field.accessors());
        let builder = OperationBuilder::new(self);
        let builder_tokens = builder.builder();
        let builder_fn = builder.create_op_builder_fn();
        let default_constructor = builder.default_constructor();
        let summary = &self.summary;
        let description =
            sanitize_documentation(&self.description).expect("valid Markdown documentation");

        tokens.append_all(quote! {
            #[doc = #summary]
            #[doc = "\n\n"]
            #[doc = #description]
            pub struct #class_name<'c> {
                operation: ::melior::ir::operation::Operation<'c>,
            }

            impl<'c> #class_name<'c> {
                pub fn name() -> &'static str {
                    #name
                }

                pub fn operation(&self) -> &::melior::ir::operation::Operation<'c> {
                    &self.operation
                }

                #builder_fn

                #(#accessors)*
            }

            #builder_tokens

            #default_constructor

            impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for #class_name<'c> {
                type Error = ::melior::Error;

                fn try_from(
                    operation: ::melior::ir::operation::Operation<'c>,
                ) -> Result<Self, Self::Error> {
                    Ok(Self { operation })
                }
            }

            impl<'c> Into<::melior::ir::operation::Operation<'c>> for #class_name<'c> {
                fn into(self) -> ::melior::ir::operation::Operation<'c> {
                    self.operation
                }
            }
        })
    }
}
