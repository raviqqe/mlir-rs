mod accessors;
mod builder;
mod element_kind;
mod field_kind;
mod operation_field;
mod sequence_info;
mod variadic_kind;

use self::{
    builder::{generate_operation_builder, OperationBuilder},
    element_kind::ElementKind,
    field_kind::FieldKind,
    operation_field::OperationField,
    sequence_info::SequenceInfo,
    variadic_kind::VariadicKind,
};
use super::utility::sanitize_documentation;
use crate::dialect::{
    error::{Error, OdsError},
    types::{AttributeConstraint, RegionConstraint, SuccessorConstraint, Trait, TypeConstraint},
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use tblgen::{error::WithLocation, record::Record};

pub fn generate_operation(operation: &Operation) -> Result<TokenStream, Error> {
    let summary = operation.summary()?;
    let description = operation.description()?;
    let class_name = format_ident!("{}", operation.class_name()?);
    let name = &operation.full_name()?;
    let accessors = operation
        .fields()
        .map(|field| field.accessors())
        .collect::<Result<Vec<_>, _>>()?;

    let builder = OperationBuilder::new(operation)?;
    let builder_tokens = generate_operation_builder(&builder)?;
    let builder_fn = builder.create_op_builder_fn()?;
    let default_constructor = builder.create_default_constructor()?;

    Ok(quote! {
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
                // TODO Check an operation name.
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

#[derive(Debug)]
pub struct Operation<'a> {
    definition: Record<'a>,
    can_infer_type: bool,
    regions: Vec<OperationField<'a>>,
    successors: Vec<OperationField<'a>>,
    results: Vec<OperationField<'a>>,
    operands: Vec<OperationField<'a>>,
    attributes: Vec<OperationField<'a>>,
    derived_attributes: Vec<OperationField<'a>>,
}

impl<'a> Operation<'a> {
    pub fn new(definition: Record<'a>) -> Result<Self, Error> {
        let traits = Self::collect_traits(definition)?;
        let has_trait = |name| traits.iter().any(|r#trait| r#trait.name() == Some(name));

        let arguments = Self::dag_constraints(definition, "arguments")?;
        let regions = Self::collect_regions(definition)?;
        let (results, unfixed_result_count) = Self::collect_results(
            definition,
            has_trait("::mlir::OpTrait::SameVariadicResultSize"),
            has_trait("::mlir::OpTrait::AttrSizedResultSegments"),
        )?;

        Ok(Self {
            successors: Self::collect_successors(definition)?,
            operands: Self::collect_operands(
                &arguments,
                has_trait("::mlir::OpTrait::SameVariadicOperandSize"),
                has_trait("::mlir::OpTrait::AttrSizedOperandSegments"),
            )?,
            results,
            attributes: Self::collect_attributes(&arguments)?,
            derived_attributes: Self::collect_derived_attributes(definition)?,
            can_infer_type: traits.iter().any(|r#trait| {
                (r#trait.name() == Some("::mlir::OpTrait::FirstAttrDerivedResultType")
                    || r#trait.name() == Some("::mlir::OpTrait::SameOperandsAndResultType"))
                    && unfixed_result_count == 0
                    || r#trait.name() == Some("::mlir::InferTypeOpInterface::Trait")
                        && regions.is_empty()
            }),
            regions,
            definition,
        })
    }

    fn dialect(&self) -> Result<Record, Error> {
        Ok(self.definition.def_value("opDialect")?)
    }

    pub fn dialect_name(&self) -> Result<&str, Error> {
        Ok(self.dialect()?.name()?)
    }

    pub fn class_name(&self) -> Result<&str, Error> {
        let name = self.definition.name()?;

        Ok(if name.starts_with('_') {
            name
        } else if let Some(name) = name.split('_').nth(1) {
            // Trim dialect prefix from name.
            name
        } else {
            name
        })
    }

    pub fn short_name(&self) -> Result<&str, Error> {
        Ok(self.definition.str_value("opName")?)
    }

    pub fn full_name(&self) -> Result<String, Error> {
        let dialect_name = self.dialect()?.string_value("name")?;
        let short_name = self.short_name()?;

        Ok(if dialect_name.is_empty() {
            short_name.into()
        } else {
            format!("{dialect_name}.{short_name}")
        })
    }

    pub fn summary(&self) -> Result<String, Error> {
        let short_name = self.short_name()?;
        let class_name = self.class_name()?;
        let summary = self.definition.str_value("summary")?;

        Ok([
            format!("[`{short_name}`]({class_name}) operation."),
            if summary.is_empty() {
                Default::default()
            } else {
                summary[0..1].to_uppercase() + &summary[1..] + "."
            },
        ]
        .join(" "))
    }

    pub fn description(&self) -> Result<String, Error> {
        sanitize_documentation(self.definition.str_value("description")?)
    }

    pub fn fields(&self) -> impl Iterator<Item = &OperationField<'a>> + Clone {
        self.results
            .iter()
            .chain(&self.operands)
            .chain(&self.regions)
            .chain(&self.successors)
            .chain(&self.attributes)
            .chain(&self.derived_attributes)
    }

    fn collect_successors(definition: Record<'a>) -> Result<Vec<OperationField>, Error> {
        let successors_dag = definition.dag_value("successors")?;
        let len = successors_dag.num_args();

        successors_dag
            .args()
            .enumerate()
            .map(|(index, (name, value))| {
                OperationField::new_successor(
                    name,
                    SuccessorConstraint::new(
                        value
                            .try_into()
                            .map_err(|error: tblgen::Error| error.set_location(definition))?,
                    ),
                    SequenceInfo { index, len },
                )
            })
            .collect()
    }

    fn collect_regions(definition: Record<'a>) -> Result<Vec<OperationField>, Error> {
        let regions_dag = definition.dag_value("regions")?;
        let len = regions_dag.num_args();

        regions_dag
            .args()
            .enumerate()
            .map(|(index, (name, value))| {
                OperationField::new_region(
                    name,
                    RegionConstraint::new(
                        value
                            .try_into()
                            .map_err(|error: tblgen::Error| error.set_location(definition))?,
                    ),
                    SequenceInfo { index, len },
                )
            })
            .collect()
    }

    fn collect_traits(definition: Record<'a>) -> Result<Vec<Trait>, Error> {
        let mut trait_lists = vec![definition.list_value("traits")?];
        let mut traits = vec![];

        while let Some(trait_list) = trait_lists.pop() {
            for value in trait_list.iter() {
                let definition: Record = value
                    .try_into()
                    .map_err(|error: tblgen::Error| error.set_location(definition))?;

                if definition.subclass_of("TraitList") {
                    trait_lists.push(definition.list_value("traits")?);
                } else {
                    if definition.subclass_of("Interface") {
                        trait_lists.push(definition.list_value("baseInterfaces")?);
                    }
                    traits.push(Trait::new(definition)?)
                }
            }
        }

        Ok(traits)
    }

    fn dag_constraints(
        definition: Record<'a>,
        dag_field_name: &str,
    ) -> Result<Vec<(&'a str, Record<'a>)>, Error> {
        definition
            .dag_value(dag_field_name)?
            .args()
            .map(|(name, argument)| {
                let mut definition: Record = argument
                    .try_into()
                    .map_err(|error: tblgen::Error| error.set_location(definition))?;

                if definition.subclass_of("OpVariable") {
                    definition = definition.def_value("constraint")?;
                }

                Ok((name, definition))
            })
            .collect()
    }

    fn collect_results(
        def: Record<'a>,
        same_size: bool,
        attribute_sized: bool,
    ) -> Result<(Vec<OperationField>, usize), Error> {
        Self::collect_elements(
            &Self::dag_constraints(def, "results")?
                .into_iter()
                .map(|(name, constraint)| (name, TypeConstraint::new(constraint)))
                .collect::<Vec<_>>(),
            ElementKind::Result,
            same_size,
            attribute_sized,
        )
    }

    fn collect_operands(
        arguments: &[(&'a str, Record<'a>)],
        same_size: bool,
        attribute_sized: bool,
    ) -> Result<Vec<OperationField<'a>>, Error> {
        Ok(Self::collect_elements(
            &arguments
                .iter()
                .filter(|(_, definition)| definition.subclass_of("TypeConstraint"))
                .map(|(name, definition)| (*name, TypeConstraint::new(*definition)))
                .collect::<Vec<_>>(),
            ElementKind::Operand,
            same_size,
            attribute_sized,
        )?
        .0)
    }

    fn collect_elements(
        elements: &[(&'a str, TypeConstraint<'a>)],
        element_kind: ElementKind,
        same_size: bool,
        attribute_sized: bool,
    ) -> Result<(Vec<OperationField<'a>>, usize), Error> {
        let unfixed_count = elements
            .iter()
            .filter(|(_, constraint)| constraint.has_unfixed())
            .count();
        let mut variadic_kind = VariadicKind::new(unfixed_count, same_size, attribute_sized);
        let mut fields = vec![];

        for (index, (name, constraint)) in elements.iter().enumerate() {
            fields.push(OperationField::new_element(
                name,
                *constraint,
                element_kind,
                SequenceInfo {
                    index,
                    len: elements.len(),
                },
                variadic_kind.clone(),
            )?);

            match &mut variadic_kind {
                VariadicKind::Simple { unfixed_seen } => {
                    if constraint.has_unfixed() {
                        *unfixed_seen = true;
                    }
                }
                VariadicKind::SameSize {
                    preceding_simple_count,
                    preceding_variadic_count,
                    ..
                } => {
                    if constraint.has_unfixed() {
                        *preceding_variadic_count += 1;
                    } else {
                        *preceding_simple_count += 1;
                    }
                }
                VariadicKind::AttributeSized => {}
            }
        }

        Ok((fields, unfixed_count))
    }

    fn collect_attributes(
        arguments: &[(&'a str, Record<'a>)],
    ) -> Result<Vec<OperationField<'a>>, Error> {
        arguments
            .iter()
            .filter(|(_, definition)| definition.subclass_of("Attr"))
            .map(|(name, definition)| {
                if definition.subclass_of("DerivedAttr") {
                    Err(OdsError::UnexpectedSuperClass("DerivedAttr")
                        .with_location(*definition)
                        .into())
                } else {
                    OperationField::new_attribute(name, AttributeConstraint::new(*definition))
                }
            })
            .collect()
    }

    fn collect_derived_attributes(
        definition: Record<'a>,
    ) -> Result<Vec<OperationField<'a>>, Error> {
        definition
            .values()
            .filter_map(|value| {
                let Ok(def) = Record::try_from(value) else {
                    return None;
                };
                def.subclass_of("Attr").then_some(def)
            })
            .map(|definition| {
                if definition.subclass_of("DerivedAttr") {
                    OperationField::new_attribute(
                        definition.name()?,
                        AttributeConstraint::new(definition),
                    )
                } else {
                    Err(OdsError::ExpectedSuperClass("DerivedAttr")
                        .with_location(definition)
                        .into())
                }
            })
            .collect()
    }
}
