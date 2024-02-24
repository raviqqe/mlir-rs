use quote::format_ident;
use std::iter::repeat;
use syn::{parse_quote, GenericArgument};

const RESULT_PREFIX: &str = "T";
const OPERAND_PREFIX: &str = "O";
const REGION_PREFIX: &str = "R";
const SUCCESSOR_PREFIX: &str = "S";
const ATTRIBUTE_PREFIX: &str = "A";

#[derive(Debug)]
pub struct TypeState {
    results: Vec<String>,
    operands: Vec<String>,
    regions: Vec<String>,
    successors: Vec<String>,
    attributes: Vec<String>,
}

impl TypeState {
    pub fn new(
        results: Vec<String>,
        operands: Vec<String>,
        regions: Vec<String>,
        successors: Vec<String>,
        attributes: Vec<String>,
    ) -> Self {
        Self {
            results,
            operands,
            regions,
            successors,
            attributes,
        }
    }

    pub fn parameters(&self) -> impl Iterator<Item = GenericArgument> {
        self.results()
            .chain(self.operands())
            .chain(self.regions())
            .chain(self.successors())
            .chain(self.attributes())
    }

    fn results(&self) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters(&self.results, RESULT_PREFIX)
    }

    fn operands(&self) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters(&self.operands, OPERAND_PREFIX)
    }

    fn regions(&self) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters(&self.regions, REGION_PREFIX)
    }

    fn successors(&self) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters(&self.successors, SUCCESSOR_PREFIX)
    }

    fn attributes(&self) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters(&self.attributes, ATTRIBUTE_PREFIX)
    }

    pub fn parameters_without<'a>(
        &'a self,
        field: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        self.results_without(field)
            .chain(self.operands_without(field))
            .chain(self.regions_without(field))
            .chain(self.successors_without(field))
            .chain(self.attributes_without(field))
    }

    fn results_without<'a>(&'a self, field: &'a str) -> impl Iterator<Item = GenericArgument> + 'a {
        Self::build_parameters_without(&self.results, RESULT_PREFIX, field)
    }

    fn operands_without<'a>(
        &'a self,
        field: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        Self::build_parameters_without(&self.operands, OPERAND_PREFIX, field)
    }

    fn regions_without<'a>(&'a self, field: &'a str) -> impl Iterator<Item = GenericArgument> + 'a {
        Self::build_parameters_without(&self.regions, REGION_PREFIX, field)
    }

    fn successors_without<'a>(
        &'a self,
        field: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        Self::build_parameters_without(&self.successors, SUCCESSOR_PREFIX, field)
    }

    fn attributes_without<'a>(
        &'a self,
        field: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        Self::build_parameters_without(&self.attributes, ATTRIBUTE_PREFIX, field)
    }

    pub fn arguments_with<'a>(
        &'a self,
        field: &'a str,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        self.results_with(field, set)
            .chain(self.operands_with(field, set))
            .chain(self.regions_with(field, set))
            .chain(self.successors_with(field, set))
            .chain(self.attributes_with(field, set))
    }

    fn results_with<'a>(
        &'a self,
        field: &'a str,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        Self::build_arguments_with(&self.results, RESULT_PREFIX, field, set)
    }

    fn operands_with<'a>(
        &'a self,
        field: &'a str,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        Self::build_arguments_with(&self.operands, OPERAND_PREFIX, field, set)
    }

    fn regions_with<'a>(
        &'a self,
        field: &'a str,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        Self::build_arguments_with(&self.regions, REGION_PREFIX, field, set)
    }

    fn successors_with<'a>(
        &'a self,
        field: &'a str,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        Self::build_arguments_with(&self.successors, SUCCESSOR_PREFIX, field, set)
    }

    fn attributes_with<'a>(
        &'a self,
        field: &'a str,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        Self::build_arguments_with(&self.attributes, ATTRIBUTE_PREFIX, field, set)
    }

    pub fn arguments_with_all(&self, set: bool) -> impl Iterator<Item = GenericArgument> {
        self.results_with_all(set)
            .chain(self.operands_with_all(set))
            .chain(self.regions_with_all(set))
            .chain(self.successors_with_all(set))
            .chain(self.attributes_with_all(set))
    }

    fn results_with_all(&self, set: bool) -> impl Iterator<Item = GenericArgument> {
        Self::build_arguments_with_all(&self.results, set)
    }

    fn operands_with_all(&self, set: bool) -> impl Iterator<Item = GenericArgument> {
        Self::build_arguments_with_all(&self.operands, set)
    }

    fn regions_with_all(&self, set: bool) -> impl Iterator<Item = GenericArgument> {
        Self::build_arguments_with_all(&self.regions, set)
    }

    fn successors_with_all(&self, set: bool) -> impl Iterator<Item = GenericArgument> {
        Self::build_arguments_with_all(&self.successors, set)
    }

    fn attributes_with_all(&self, set: bool) -> impl Iterator<Item = GenericArgument> {
        Self::build_arguments_with_all(&self.attributes, set)
    }

    fn build_parameters<'a>(
        fields: &[String],
        prefix: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        (0..fields.len()).map(|index| Self::build_generic_argument(prefix, index))
    }

    fn build_parameters_without<'a>(
        fields: &'a [String],
        prefix: &'a str,
        field: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        // let index = fields
        //     .iter()
        //     .position(|other| other == field)
        //     .unwrap_or(fields.len());

        // repeat(Self::build_argument(true))
        //     .take(index)
        //     .chain(repeat(Self::build_argument(false)).take(fields.len() - index));

        fields
            .iter()
            .enumerate()
            .filter(move |(_, other)| *other != field)
            .map(|(index, _)| Self::build_generic_argument(prefix, index))
    }

    fn build_arguments_with<'a>(
        fields: &'a [String],
        prefix: &'a str,
        field: &'a str,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        fields.iter().enumerate().map(move |(index, other)| {
            if other == field {
                Self::build_argument(set)
            } else {
                Self::build_generic_argument(prefix, index)
            }
        })
    }

    fn build_arguments_with_all(
        fields: &[String],
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> {
        repeat(Self::build_argument(set)).take(fields.len())
    }

    fn build_generic_argument(prefix: &str, index: usize) -> GenericArgument {
        let identifier = format_ident!("{prefix}{index}");

        parse_quote!(#identifier)
    }

    fn build_argument(set: bool) -> GenericArgument {
        if set {
            parse_quote!(::melior::dialect::ods::__private::Set)
        } else {
            parse_quote!(::melior::dialect::ods::__private::Unset)
        }
    }
}
