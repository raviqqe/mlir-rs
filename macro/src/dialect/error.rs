use proc_macro2::Span;
use std::{
    error,
    fmt::{self, Display, Formatter},
};
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

impl error::Error for Error {}

impl From<SourceError<ExpectedSuperClassError>> for Error {
    fn from(error: SourceError<ExpectedSuperClassError>) -> Self {
        Self::ExpectedSuperClass(error)
    }
}

impl From<SourceError<TableGenError>> for Error {
    fn from(error: SourceError<TableGenError>) -> Self {
        Self::TableGen(error)
    }
}

impl From<syn::Error> for Error {
    fn from(error: syn::Error) -> Self {
        Self::Syn(error)
    }
}

impl From<Error> for syn::Error {
    fn from(error: Error) -> Self {
        match error {
            Error::Syn(error) => error,
            _ => syn::Error::new(Span::call_site(), format!("{}", error)),
        }
    }
}

#[derive(Debug)]
pub struct ExpectedSuperClassError(pub String);

impl Display for ExpectedSuperClassError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        write!(
            formatter,
            "expected this record to be a subclass of {}",
            self.0
        )
    }
}

impl error::Error for ExpectedSuperClassError {}
