use proc_macro2::Span;
use std::fmt::{self, Display, Formatter};
use tblgen::{
    error::{SourceError, TableGenError},
    SourceInfo,
};

#[derive(Debug)]
pub enum Error {
    Syn(syn::Error),
    TableGen(tblgen::Error),
    ExpectedSuperClass(SourceError<ExpectedSuperClassError>),
    ParseError,
}

impl Error {
    pub fn add_source_info(self, info: SourceInfo) -> Self {
        match self {
            Self::TableGen(error) => error.add_source_info(info).into(),
            Self::ExpectedSuperClass(error) => error.add_source_info(info).into(),
            Self::Syn(_) | Self::ParseError => self,
        }
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::Syn(error) => write!(formatter, "failed to parse macro input: {error}"),
            Self::TableGen(error) => write!(formatter, "invalid ODS input: {error}"),
            Self::ExpectedSuperClass(error) => write!(formatter, "invalid ODS input: {error}"),
            Self::ParseError => write!(formatter, "error parsing TableGen source"),
        }
    }
}

impl std::error::Error for Error {}

impl From<SourceError<ExpectedSuperClassError>> for Error {
    fn from(value: SourceError<ExpectedSuperClassError>) -> Self {
        Self::ExpectedSuperClass(value)
    }
}

impl From<SourceError<TableGenError>> for Error {
    fn from(value: SourceError<TableGenError>) -> Self {
        Self::TableGen(value)
    }
}

impl From<syn::Error> for Error {
    fn from(value: syn::Error) -> Self {
        Self::Syn(value)
    }
}

impl From<Error> for syn::Error {
    fn from(value: Error) -> Self {
        match value {
            Error::Syn(e) => e,
            _ => syn::Error::new(Span::call_site(), format!("{}", value)),
        }
    }
}

#[derive(Debug)]
pub struct ExpectedSuperClassError(pub String);

impl Display for ExpectedSuperClassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expected this record to be a subclass of {}", self.0)
    }
}

impl std::error::Error for ExpectedSuperClassError {}
