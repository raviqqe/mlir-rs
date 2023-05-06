use std::{
    error,
    fmt::{self, Display, Formatter},
};

/// A Melior error.
#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    BlockArgumentExpected(String),
    BlockArgumentPosition(String, usize),
    FunctionExpected(String),
    FunctionInputPosition(String, usize),
    FunctionResultPosition(String, usize),
    InvokeFunction,
    MemRefExpected(String),
    OperationResultExpected(String),
    OperationResultPosition(String, usize),
    ParsePassPipeline(String),
    RunPass,
    TupleExpected(String),
    TupleFieldPosition(String, usize),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::BlockArgumentExpected(value) => {
                write!(formatter, "block argument expected: {value}")
            }
            Self::BlockArgumentPosition(block, position) => {
                write!(
                    formatter,
                    "block argument position {position} out of range: {block}"
                )
            }
            Self::FunctionExpected(r#type) => write!(formatter, "function expected: {type}"),
            Self::FunctionInputPosition(r#type, position) => write!(
                formatter,
                "function input position {position} out of range: {type}"
            ),
            Self::FunctionResultPosition(r#type, position) => write!(
                formatter,
                "function result position {position} out of range: {type}"
            ),
            Self::InvokeFunction => write!(formatter, "failed to invoke JIT-compiled function"),
            Self::MemRefExpected(r#type) => write!(formatter, "mem-ref expected: {type}"),
            Self::OperationResultExpected(value) => {
                write!(formatter, "operation result expected: {value}")
            }
            Self::OperationResultPosition(operation, position) => {
                write!(
                    formatter,
                    "operation result position {position} out of range: {operation}"
                )
            }
            Self::ParsePassPipeline(message) => {
                write!(formatter, "failed to parse pass pipeline: {}", message)
            }
            Self::RunPass => write!(formatter, "failed to run pass"),
            Self::TupleExpected(r#type) => write!(formatter, "tuple expected: {type}"),
            Self::TupleFieldPosition(r#type, position) => {
                write!(
                    formatter,
                    "tuple field position {position} out of range: {type}"
                )
            }
        }
    }
}

impl error::Error for Error {}
