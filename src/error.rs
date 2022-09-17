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
    InvokeFunction,
    OperationResultExpected(String),
    OperationResultPosition(String, usize),
    RunPass,
    ParsePassPipeline,
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::BlockArgumentExpected(value) => {
                write!(formatter, "block argument expected: {}", value)
            }
            Self::BlockArgumentPosition(block, position) => {
                write!(
                    formatter,
                    "block argument position {} out of range: {}",
                    position, block
                )
            }
            Self::FunctionExpected(r#type) => write!(formatter, "function expected: {}", r#type),
            Self::InvokeFunction => write!(formatter, "failed to invoke JIT-compiled function"),
            Self::OperationResultExpected(value) => {
                write!(formatter, "operation result expected: {}", value)
            }
            Self::OperationResultPosition(operation, position) => {
                write!(
                    formatter,
                    "operation result position {} out of range: {}",
                    position, operation
                )
            }
            Self::RunPass => write!(formatter, "failed to run pass"),
            Self::ParsePassPipeline => write!(formatter, "failed to parse pass pipeline"),
        }
    }
}

impl error::Error for Error {}
