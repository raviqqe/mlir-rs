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
    ParsePassPipeline(String),
    RunPass,
    TupleFieldPosition(String, usize),
    TypeExpected(&'static str, String),
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
            Self::BlockArgumentPosition(block, position) => {
                write!(
                    formatter,
                    "block argument position {position} out of bounds: {block}"
                )
            }
            Self::FunctionInputPosition(r#type, position) => write!(
                formatter,
                "function input position {position} out of bounds: {type}"
            ),
            Self::FunctionResultPosition(r#type, position) => write!(
                formatter,
                "function result position {position} out of bounds: {type}"
            ),
            Self::InvokeFunction => write!(formatter, "failed to invoke JIT-compiled function"),
            Self::OperationResultExpected(value) => {
                write!(formatter, "operation result expected: {value}")
            }
            Self::OperationResultPosition(operation, position) => {
                write!(
                    formatter,
                    "operation result position {position} out of bounds: {operation}"
                )
            }
            Self::ParsePassPipeline(message) => {
                write!(formatter, "failed to parse pass pipeline:\n{}", message)
            }
            Self::RunPass => write!(formatter, "failed to run pass"),
            Self::TupleFieldPosition(r#type, position) => {
                write!(
                    formatter,
                    "tuple field position {position} out of bounds: {type}"
                )
            }
            Self::TypeExpected(r#type, actual) => {
                write!(formatter, "{type} type expected: {actual}")
            }
        }
    }
}

impl error::Error for Error {}
