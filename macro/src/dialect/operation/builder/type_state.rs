use quote::format_ident;
use std::iter::repeat;
use syn::{parse_quote, GenericArgument};

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

    pub fn results(&self) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters(&self.results, "T")
    }

    pub fn operands(&self) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters(&self.operands, "O")
    }

    pub fn regions(&self) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters(&self.regions, "R")
    }

    pub fn successors(&self) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters(&self.successors, "S")
    }

    pub fn attributes(&self) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters(&self.attributes, "A")
    }

    pub fn results_without(&self, field: &str) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters_without(&self.results, field)
    }

    pub fn operands_without(&self, field: &str) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters_without(&self.operands, field)
    }

    pub fn regions_without(&self, field: &str) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters_without(&self.regions, field)
    }

    pub fn successors_without(&self, field: &str) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters_without(&self.successors, field)
    }

    pub fn attributes_without(&self, field: &str) -> impl Iterator<Item = GenericArgument> {
        Self::build_parameters_without(&self.attributes, field)
    }

    pub fn results_with_all(&self, set: bool) -> impl Iterator<Item = GenericArgument> {
        Self::build_arguments_with_all(&self.results, set)
    }

    pub fn operands_with_all(&self, set: bool) -> impl Iterator<Item = GenericArgument> {
        Self::build_arguments_with_all(&self.operands, set)
    }

    pub fn regions_with_all(&self, set: bool) -> impl Iterator<Item = GenericArgument> {
        Self::build_arguments_with_all(&self.regions, set)
    }

    pub fn successors_with_all(&self, set: bool) -> impl Iterator<Item = GenericArgument> {
        Self::build_arguments_with_all(&self.successors, set)
    }

    pub fn attributes_with_all(&self, set: bool) -> impl Iterator<Item = GenericArgument> {
        Self::build_arguments_with_all(&self.attributes, set)
    }

    fn build_parameters<'a>(
        fields: &[String],
        prefix: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        (0..fields.len()).map(|index| Self::build_generic_argument(prefix, index))
    }

    fn build_parameters_without(
        fields: &[String],
        field: &str,
    ) -> impl Iterator<Item = GenericArgument> {
        let index = fields.iter().position(|other| other == field).unwrap();

        repeat(Self::build_argument(true))
            .take(index)
            .chain(repeat(Self::build_argument(false)).take(fields.len() - index))
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
