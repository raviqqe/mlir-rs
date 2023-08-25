mod accessors;
mod builder;

use self::builder::OperationBuilder;
use super::utility::{sanitize_documentation, sanitize_snake_case_name};
use crate::dialect::{
    error::{Error, OdsError},
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

    pub fn param_type(&self) -> Result<Type, Error> {
        Ok(match self {
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
                    let r#type: Type = syn::parse_str(constraint.storage_type()?)
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
        })
    }

    fn create_result_type(r#type: Type) -> Type {
        parse_quote!(Result<#r#type, ::melior::Error>)
    }

    fn create_iterator_type(r#type: Type) -> Type {
        parse_quote!(impl Iterator<Item = #r#type>)
    }

    pub fn return_type(&self) -> Result<Type, Error> {
        Ok(match self {
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
                    Self::create_result_type(self.param_type()?)
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
        })
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

impl VariadicKind {
    pub fn new(num_variable_length: usize, same_size: bool, attr_sized: bool) -> Self {
        if num_variable_length <= 1 {
            VariadicKind::Simple {
                seen_variable_length: false,
            }
        } else if same_size {
            VariadicKind::SameSize {
                num_variable_length,
                num_preceding_simple: 0,
                num_preceding_variadic: 0,
            }
        } else if attr_sized {
            VariadicKind::AttrSized {}
        } else {
            unimplemented!()
        }
    }
}

pub struct VariadicKindIter<I> {
    current: VariadicKind,
    iter: I,
}

impl<'a: 'b, 'b, I: Iterator<Item = &'b TypeConstraint<'a>>> VariadicKindIter<I> {
    pub fn new(iter: I, num_variable_length: usize, same_size: bool, attr_sized: bool) -> Self {
        Self {
            iter,
            current: VariadicKind::new(num_variable_length, same_size, attr_sized),
        }
    }
}

impl<'a: 'b, 'b, I: Iterator<Item = &'b TypeConstraint<'a>>> Iterator for VariadicKindIter<I> {
    type Item = VariadicKind;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(constraint) = self.iter.next() else {
            return None;
        };
        let original = self.current.clone();
        match &mut self.current {
            VariadicKind::Simple {
                seen_variable_length,
            } => {
                if constraint.has_variable_length() {
                    *seen_variable_length = true;
                }
            }
            VariadicKind::SameSize {
                num_preceding_simple,
                num_preceding_variadic,
                ..
            } => {
                if constraint.has_variable_length() {
                    *num_preceding_variadic += 1;
                } else {
                    *num_preceding_simple += 1;
                }
            }
            VariadicKind::AttrSized {} => {}
        };
        Some(original)
    }
}

#[derive(Debug, Clone)]
pub struct OperationField<'a> {
    pub(crate) name: &'a str,
    pub(crate) sanitized_name: Ident,
    pub(crate) kind: FieldKind<'a>,
}

impl<'a> OperationField<'a> {
    fn new(name: &'a str, kind: FieldKind<'a>) -> Self {
        Self {
            name,
            sanitized_name: sanitize_snake_case_name(name),
            kind,
        }
    }

    fn new_attribute(name: &'a str, constraint: AttributeConstraint<'a>) -> Self {
        Self::new(name, FieldKind::Attribute { constraint })
    }

    fn new_region(
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

    fn new_successor(
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
    pub(crate) regions: Vec<OperationField<'a>>,
    pub(crate) successors: Vec<OperationField<'a>>,
    pub(crate) results: Vec<OperationField<'a>>,
    pub(crate) operands: Vec<OperationField<'a>>,
    pub(crate) attributes: Vec<OperationField<'a>>,
    pub(crate) derived_attributes: Vec<OperationField<'a>>,
    pub(crate) can_infer_type: bool,
    pub(crate) summary: String,
    pub(crate) description: String,
}

impl<'a> Operation<'a> {
    pub fn fields(&self) -> impl Iterator<Item = &OperationField<'a>> + Clone {
        self.results
            .iter()
            .chain(self.operands.iter())
            .chain(self.regions.iter())
            .chain(self.successors.iter())
            .chain(self.attributes.iter())
            .chain(self.derived_attributes.iter())
    }

    fn collect_successors(def: Record<'a>) -> Result<Vec<OperationField>, Error> {
        let successors_dag = def.dag_value("successors")?;
        let len = successors_dag.num_args();
        successors_dag
            .args()
            .enumerate()
            .map(|(index, (name, value))| {
                Ok(OperationField::new_successor(
                    name,
                    SuccessorConstraint::new(
                        value
                            .try_into()
                            .map_err(|e: tblgen::Error| e.set_location(def))?,
                    ),
                    SequenceInfo { index, len },
                ))
            })
            .collect()
    }

    fn collect_regions(def: Record<'a>) -> Result<Vec<OperationField>, Error> {
        let regions_dag = def.dag_value("regions")?;
        let len = regions_dag.num_args();
        regions_dag
            .args()
            .enumerate()
            .map(|(index, (name, v))| {
                Ok(OperationField::new_region(
                    name,
                    RegionConstraint::new(
                        v.try_into()
                            .map_err(|e: tblgen::Error| e.set_location(def))?,
                    ),
                    SequenceInfo { index, len },
                ))
            })
            .collect()
    }

    fn collect_traits(def: Record<'a>) -> Result<Vec<Trait>, Error> {
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
        Ok(traits)
    }

    fn dag_constraints(
        def: Record<'a>,
        dag_field_name: &str,
    ) -> Result<Vec<(&'a str, Record<'a>)>, Error> {
        def.dag_value(dag_field_name)?
            .args()
            .map(|(name, arg)| {
                let mut arg_def: Record = arg
                    .try_into()
                    .map_err(|e: tblgen::Error| e.set_location(def))?;

                if arg_def.subclass_of("OpVariable") {
                    arg_def = arg_def.def_value("constraint")?;
                }

                Ok((name, arg_def))
            })
            .collect()
    }

    fn collect_results(
        def: Record<'a>,
        same_size: bool,
        attr_sized: bool,
    ) -> Result<(Vec<OperationField>, usize), Error> {
        Self::collect_elements(
            Self::dag_constraints(def, "results")?
                .into_iter()
                .map(|(name, constraint)| (name, TypeConstraint::new(constraint)))
                .collect::<Vec<_>>()
                .iter(),
            ElementKind::Result,
            same_size,
            attr_sized,
        )
    }

    fn collect_operands<'b, 'c: 'a + 'b>(
        arguments: impl Iterator<Item = &'b (&'c str, Record<'c>)>,
        same_size: bool,
        attr_sized: bool,
    ) -> Result<Vec<OperationField<'a>>, Error> {
        Ok(Self::collect_elements(
            arguments
                .filter(|(_, arg_def)| arg_def.subclass_of("TypeConstraint"))
                .map(|(name, arg_def)| (*name, TypeConstraint::new(*arg_def)))
                .collect::<Vec<_>>()
                .iter(),
            ElementKind::Operand,
            same_size,
            attr_sized,
        )?
        .0)
    }

    fn collect_elements<'b, 'c: 'a + 'b>(
        elements: impl Iterator<Item = &'b (&'c str, TypeConstraint<'c>)> + Clone,
        kind: ElementKind,
        same_size: bool,
        attr_sized: bool,
    ) -> Result<(Vec<OperationField<'a>>, usize), Error> {
        let len = elements.clone().count();
        let num_variable_length = elements
            .clone()
            .filter(|res| res.1.has_variable_length())
            .count();
        let variadic_iter = VariadicKindIter::new(
            elements.clone().map(|(_, tc)| tc),
            num_variable_length,
            same_size,
            attr_sized,
        );
        Ok((
            elements
                .enumerate()
                .zip(variadic_iter)
                .map(|((index, (name, constraint)), variadic_kind)| {
                    OperationField::new_element(
                        name,
                        *constraint,
                        kind,
                        SequenceInfo { index, len },
                        variadic_kind,
                    )
                })
                .collect::<Vec<_>>(),
            num_variable_length,
        ))
    }

    fn collect_attributes<'b, 'c: 'a + 'b>(
        arguments: impl Iterator<Item = &'b (&'c str, Record<'c>)>,
    ) -> Result<Vec<OperationField<'a>>, Error> {
        arguments
            .filter(|(_, arg_def)| arg_def.subclass_of("Attr"))
            .map(|(name, arg_def)| {
                // TODO: Replace assert! with Result
                assert!(!name.is_empty());
                assert!(!arg_def.subclass_of("DerivedAttr"));
                Ok(OperationField::new_attribute(
                    name,
                    AttributeConstraint::new(*arg_def),
                ))
            })
            .collect()
    }

    fn collect_derived_attributes(def: Record<'a>) -> Result<Vec<OperationField<'a>>, Error> {
        def.values()
            .filter_map(|value| {
                let Ok(def) = Record::try_from(value) else {
                    return None;
                };
                def.subclass_of("Attr").then_some(def)
            })
            .map(|def| {
                def.subclass_of("DerivedAttr")
                    .then_some(())
                    .ok_or_else(|| {
                        OdsError::ExpectedSuperClass("DerivedAttr").with_location(def)
                    })?;
                Ok(OperationField::new_attribute(
                    def.name()?,
                    AttributeConstraint::new(def),
                ))
            })
            .collect()
    }

    pub fn from_def(def: Record<'a>) -> Result<Self, Error> {
        let dialect = def.def_value("opDialect")?;
        let traits = Self::collect_traits(def)?;
        let has_trait = |name: &str| traits.iter().any(|r#trait| r#trait.has_name(name));

        let arguments = Self::dag_constraints(def, "arguments")?;
        let regions = Self::collect_regions(def)?;
        let (results, num_variable_length_results) = Self::collect_results(
            def,
            has_trait("::mlir::OpTrait::SameVariadicResultSize"),
            has_trait("::mlir::OpTrait::AttrSizedResultSegments"),
        )?;

        let name = def.name()?;
        let class_name = if name.contains('_') && !name.starts_with('_') {
            // Trim dialect prefix from name
            name.split('_')
                .nth(1)
                .expect("string contains separator '_'")
        } else {
            name
        };
        let short_name = def.str_value("opName")?;

        Ok(Self {
            dialect,
            short_name,
            full_name: {
                let dialect_name = dialect.string_value("name")?;

                if dialect_name.is_empty() {
                    short_name.into()
                } else {
                    format!("{dialect_name}.{short_name}")
                }
            },
            class_name,
            successors: Self::collect_successors(def)?,
            operands: Self::collect_operands(
                arguments.iter(),
                has_trait("::mlir::OpTrait::SameVariadicOperandSize"),
                has_trait("::mlir::OpTrait::AttrSizedOperandSegments"),
            )?,
            results,
            attributes: Self::collect_attributes(arguments.iter())?,
            derived_attributes: Self::collect_derived_attributes(def)?,
            can_infer_type: traits.iter().any(|r#trait| {
                (r#trait.has_name("::mlir::OpTrait::FirstAttrDerivedResultType")
                    || r#trait.has_name("::mlir::OpTrait::SameOperandsAndResultType"))
                    && num_variable_length_results == 0
                    || r#trait.has_name("::mlir::InferTypeOpInterface::Trait") && regions.is_empty()
            }),
            summary: {
                let summary = def.str_value("summary").unwrap_or(short_name);

                if summary.is_empty() {
                    format!("[`{short_name}`]({class_name}) operation")
                } else {
                    format!(
                        "[`{short_name}`]({class_name}) operation: {}",
                        summary[0..1].to_uppercase() + &summary[1..]
                    )
                }
            },
            description: unindent::unindent(def.str_value("description")?),
            regions,
        })
    }
}

impl<'a> ToTokens for Operation<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let class_name = format_ident!("{}", &self.class_name);
        let name = &self.full_name;
        let accessors = self
            .fields()
            .map(|field| field.accessors().expect("valid accessors"));
        let builder = OperationBuilder::new(self);
        let builder_tokens = builder.builder().expect("valid builder");
        let builder_fn = builder.create_op_builder_fn();
        let default_constructor = builder.default_constructor().expect("valid constructor");
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

            impl<'c> From<#class_name<'c>> for ::melior::ir::operation::Operation<'c> {
                fn from(operation: #class_name<'c>) -> ::melior::ir::operation::Operation<'c> {
                    operation.operation
                }
            }
        })
    }
}
