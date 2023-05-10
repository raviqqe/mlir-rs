use std::{
    error,
    fmt::{self, Display, Formatter},
};

/// A Melior error.
#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    ArrayElementPosition(String, usize),
    AttributeExpected(&'static str, String),
    BlockArgumentExpected(String),
    BlockArgumentPosition(String, usize),
    FunctionInputPosition(String, usize),
    FunctionResultPosition(String, usize),
    InvokeFunction,
    OperationResultExpected(String),
    OperationResultPosition(String, usize),
    PositionOutOfBounds {
        name: &'static str,
        value: String,
        index: usize,
    },
    ParsePassPipeline(String),
    RunPass,
    TupleFieldPosition(String, usize),
    TypeExpected(&'static str, String),
    UnknownDiagnosticSeverity(u32),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::ArrayElementPosition(array, position) => {
                write!(
                    formatter,
                    "array element position {position} out of bounds: {array}"
                )
            }
            Self::AttributeExpected(r#type, attribute) => {
                write!(formatter, "{type} attribute expected: {attribute}")
            }
            Self::BlockArgumentExpected(value) => {
                write!(formatter, "block argument expected: {value}")
            }
            Self::InvokeFunction => write!(formatter, "failed to invoke JIT-compiled function"),
            Self::OperationResultExpected(value) => {
                write!(formatter, "operation result expected: {value}")
            }
            Self::ParsePassPipeline(message) => {
                write!(formatter, "failed to parse pass pipeline:\n{}", message)
            }
            Self::PositionOutOfBounds { name, value, index } => {
                write!(formatter, "{name} position {index} out of bounds: {value}")
            }
            Self::RunPass => write!(formatter, "failed to run pass"),
            Self::TypeExpected(r#type, actual) => {
                write!(formatter, "{type} type expected: {actual}")
            }
            Self::UnknownDiagnosticSeverity(severity) => {
                write!(formatter, "unknown diagnostic severity: {severity}")
            }
        }
    }
}

impl error::Error for Error {}
