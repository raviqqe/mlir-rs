mod accessors;
mod builder;
mod element_kind;
mod field_kind;
mod operation_field;
mod sequence_info;
mod variadic_kind;

use self::{
    builder::OperationBuilder, element_kind::ElementKind, field_kind::FieldKind,
    operation_field::OperationField, sequence_info::SequenceInfo, variadic_kind::VariadicKind,
};
use super::utility::sanitize_documentation;
use crate::dialect::{
    error::{Error, OdsError},
    types::{AttributeConstraint, RegionConstraint, SuccessorConstraint, Trait, TypeConstraint},
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens, TokenStreamExt};
use tblgen::{error::WithLocation, record::Record};

#[derive(Clone, Debug)]
pub struct Operation<'a> {
    dialect_name: &'a str,
    short_name: &'a str,
    full_name: String,
    class_name: &'a str,
    summary: String,
    can_infer_type: bool,
    description: String,
    regions: Vec<OperationField<'a>>,
    successors: Vec<OperationField<'a>>,
    results: Vec<OperationField<'a>>,
    operands: Vec<OperationField<'a>>,
    attributes: Vec<OperationField<'a>>,
    derived_attributes: Vec<OperationField<'a>>,
}

impl<'a> Operation<'a> {
    pub fn new(definition: Record<'a>) -> Result<Self, Error> {
        let dialect = definition.def_value("opDialect")?;
        let traits = Self::collect_traits(definition)?;
        let has_trait = |name| traits.iter().any(|r#trait| r#trait.has_name(name));

        let arguments = Self::dag_constraints(definition, "arguments")?;
        let regions = Self::collect_regions(definition)?;
        let (results, variable_length_results_count) = Self::collect_results(
            definition,
            has_trait("::mlir::OpTrait::SameVariadicResultSize"),
            has_trait("::mlir::OpTrait::AttrSizedResultSegments"),
        )?;

        let name = definition.name()?;
        let class_name = if name.starts_with('_') {
            name
        } else if let Some(name) = name.split('_').nth(1) {
            // Trim dialect prefix from name.
            name
        } else {
            name
        };
        let short_name = definition.str_value("opName")?;

        Ok(Self {
            dialect_name: dialect.name()?,
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
                (r#trait.has_name("::mlir::OpTrait::FirstAttrDerivedResultType")
                    || r#trait.has_name("::mlir::OpTrait::SameOperandsAndResultType"))
                    && variable_length_results_count == 0
                    || r#trait.has_name("::mlir::InferTypeOpInterface::Trait") && regions.is_empty()
            }),
            summary: {
                let summary = definition.str_value("summary")?;

                [
                    format!("[`{short_name}`]({class_name}) operation."),
                    if summary.is_empty() {
                        Default::default()
                    } else {
                        summary[0..1].to_uppercase() + &summary[1..] + "."
                    },
                ]
                .join(" ")
            },
            description: sanitize_documentation(definition.str_value("description")?)?,
            regions,
        })
    }

    pub fn dialect_name(&self) -> &str {
        self.dialect_name
    }

    pub fn fields(&self) -> impl Iterator<Item = &OperationField<'a>> + Clone {
        self.results
            .iter()
            .chain(self.operands.iter())
            .chain(self.regions.iter())
            .chain(self.successors.iter())
            .chain(self.attributes.iter())
            .chain(self.derived_attributes.iter())
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
                let trait_definition: Record = value
                    .try_into()
                    .map_err(|error: tblgen::Error| error.set_location(definition))?;

                if trait_definition.subclass_of("TraitList") {
                    trait_lists.push(trait_definition.list_value("traits")?);
                } else {
                    if trait_definition.subclass_of("Interface") {
                        trait_lists.push(trait_definition.list_value("baseInterfaces")?);
                    }
                    traits.push(Trait::new(trait_definition)?)
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
                let mut argument_definition: Record = argument
                    .try_into()
                    .map_err(|error: tblgen::Error| error.set_location(definition))?;

                if argument_definition.subclass_of("OpVariable") {
                    argument_definition = argument_definition.def_value("constraint")?;
                }

                Ok((name, argument_definition))
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
                .filter(|(_, arg_def)| arg_def.subclass_of("TypeConstraint"))
                .map(|(name, arg_def)| (*name, TypeConstraint::new(*arg_def)))
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
        let variable_length_count = elements
            .iter()
            .filter(|(_, constraint)| constraint.has_variable_length())
            .count();
        let mut variadic_kind = VariadicKind::new(variable_length_count, same_size, attribute_sized);
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
                VariadicKind::Simple {
                    variable_length_seen: seen_variable_length,
                } => {
                    if constraint.has_variable_length() {
                        *seen_variable_length = true;
                    }
                }
                VariadicKind::SameSize {
                    preceding_simple_count,
                    preceding_variadic_count,
                    ..
                } => {
                    if constraint.has_variable_length() {
                        *preceding_variadic_count += 1;
                    } else {
                        *preceding_simple_count += 1;
                    }
                }
                VariadicKind::AttributeSized {} => {}
            }
        }

        Ok((fields, variable_length_count))
    }

    fn collect_attributes(
        arguments: &[(&'a str, Record<'a>)],
    ) -> Result<Vec<OperationField<'a>>, Error> {
        arguments
            .iter()
            .filter(|(_, arg_def)| arg_def.subclass_of("Attr"))
            .map(|(name, arg_def)| {
                // TODO: Replace assert! with Result
                assert!(!arg_def.subclass_of("DerivedAttr"));

                OperationField::new_attribute(name, AttributeConstraint::new(*arg_def))
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
                if def.subclass_of("DerivedAttr") {
                    OperationField::new_attribute(def.name()?, AttributeConstraint::new(def))
                } else {
                    Err(OdsError::ExpectedSuperClass("DerivedAttr")
                        .with_location(def)
                        .into())
                }
            })
            .collect()
    }
}

impl<'a> ToTokens for Operation<'a> {
    // TODO Compile values for proper error handling and remove `Result::expect()`.
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let class_name = format_ident!("{}", &self.class_name);
        let name = &self.full_name;
        let accessors = self
            .fields()
            .map(|field| field.accessors().expect("valid accessors"));
        let builder = OperationBuilder::new(self).expect("valid builder generator");
        let builder_tokens = builder.builder().expect("valid builder");
        let builder_fn = builder.create_op_builder_fn();
        let default_constructor = builder
            .create_default_constructor()
            .expect("valid constructor");
        let summary = &self.summary;
        let description = &self.description;

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
