mod attribute;
mod builder;
mod element_kind;
mod field_kind;
mod operation_field;
mod sequence_info;
mod variadic_kind;

pub use self::{
    attribute::Attribute, builder::OperationBuilder, element_kind::ElementKind,
    field_kind::FieldKind, operation_field::OperationField, sequence_info::SequenceInfo,
    variadic_kind::VariadicKind,
};
use super::utility::sanitize_documentation;
use crate::dialect::{
    error::{Error, OdsError},
    r#trait::Trait,
    types::{RegionConstraint, SuccessorConstraint, TypeConstraint},
};
pub use operation_field::OperationFieldLike;
use tblgen::{error::WithLocation, record::Record};

#[derive(Debug)]
pub struct Operation<'a> {
    definition: Record<'a>,
    can_infer_type: bool,
    regions: Vec<OperationField<'a>>,
    successors: Vec<OperationField<'a>>,
    results: Vec<OperationField<'a>>,
    operands: Vec<OperationField<'a>>,
    attributes: Vec<Attribute<'a>>,
    derived_attributes: Vec<Attribute<'a>>,
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

    pub fn fields(&self) -> impl Iterator<Item = &dyn OperationFieldLike> + Clone {
        self.results
            .iter()
            .chain(&self.operands)
            .chain(&self.regions)
            .chain(&self.successors)
            .map(|field| -> &dyn OperationFieldLike { field })
            .chain(
                self.attributes()
                    .map(|field| -> &dyn OperationFieldLike { field }),
            )
    }

    pub fn operation_fields(&self) -> impl Iterator<Item = &OperationField<'a>> + Clone {
        self.results
            .iter()
            .chain(&self.operands)
            .chain(&self.regions)
            .chain(&self.successors)
    }

    pub fn attributes(&self) -> impl Iterator<Item = &Attribute<'a>> + Clone {
        self.attributes.iter().chain(&self.derived_attributes)
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
                let definition =
                    Record::try_from(value).map_err(|error| error.set_location(definition))?;

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
        name: &str,
    ) -> Result<Vec<(&'a str, Record<'a>)>, Error> {
        definition
            .dag_value(name)?
            .args()
            .map(|(name, argument)| {
                Ok((
                    name,
                    if definition.subclass_of("OpVariable") {
                        definition.def_value("constraint")?
                    } else {
                        Record::try_from(argument)
                            .map_err(|error| error.set_location(definition))?
                    },
                ))
            })
            .collect()
    }

    fn collect_results(
        definition: Record<'a>,
        same_size: bool,
        attribute_sized: bool,
    ) -> Result<(Vec<OperationField>, usize), Error> {
        Self::collect_elements(
            &Self::dag_constraints(definition, "results")?
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
            .filter(|(_, constraint)| constraint.is_unfixed())
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
                    if constraint.is_unfixed() {
                        *unfixed_seen = true;
                    }
                }
                VariadicKind::SameSize {
                    preceding_simple_count,
                    preceding_variadic_count,
                    ..
                } => {
                    if constraint.is_unfixed() {
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
    ) -> Result<Vec<Attribute<'a>>, Error> {
        arguments
            .iter()
            .filter(|(_, definition)| definition.subclass_of("Attr"))
            .map(|(name, definition)| {
                if definition.subclass_of("DerivedAttr") {
                    Err(OdsError::UnexpectedSuperClass("DerivedAttr")
                        .with_location(*definition)
                        .into())
                } else {
                    Attribute::new(name, *definition)
                }
            })
            .collect()
    }

    fn collect_derived_attributes(definition: Record<'a>) -> Result<Vec<Attribute<'a>>, Error> {
        definition
            .values()
            .filter_map(|value| {
                let Ok(definition) = Record::try_from(value) else {
                    return None;
                };
                definition.subclass_of("Attr").then_some(definition)
            })
            .map(|definition| {
                if definition.subclass_of("DerivedAttr") {
                    Attribute::new(definition.name()?, definition)
                } else {
                    Err(OdsError::ExpectedSuperClass("DerivedAttr")
                        .with_location(definition)
                        .into())
                }
            })
            .collect()
    }
}
