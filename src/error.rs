use std::{
    error,
    fmt::{self, Display, Formatter},
};

/// A Melior error.
#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    BlockArgumentPosition(String, usize),
    FunctionExpected(String),
    InvokeFunction,
    OperationResultPosition(String, usize),
    ParsePassPipeline,
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::BlockArgumentPosition(block, position) => {
                write!(
                    formatter,
                    "block argument position {} out of range: {}",
                    position, block
                )
            }
            Self::FunctionExpected(r#type) => write!(formatter, "function expected: {}", r#type),
            Self::InvokeFunction => write!(formatter, "failed to invoke JIT-compiled function"),
            Self::OperationResultPosition(operation, position) => {
                write!(
                    formatter,
                    "operation result position {} out of range: {}",
                    position, operation
                )
            }
            Self::ParsePassPipeline => write!(formatter, "parse pass pipeline"),
        }
    }
}

impl error::Error for Error {}
