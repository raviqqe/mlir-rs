use crate::{location::Location, r#type::Type, utility::into_raw_array};
use mlir_sys::{mlirBlockCreate, MlirBlock};

pub struct Block {
    block: MlirBlock,
}

impl Block {
    pub fn new(arguments: Vec<(Type, Location)>) -> Self {
        Self {
            block: unsafe {
                mlirBlockCreate(
                    arguments.len() as isize,
                    into_raw_array(
                        arguments
                            .iter()
                            .map(|(argument, _)| argument.to_raw())
                            .collect(),
                    ),
                    into_raw_array(
                        arguments
                            .iter()
                            .map(|(_, location)| location.to_raw())
                            .collect(),
                    ),
                )
            },
        }
    }

    pub(crate) unsafe fn to_raw(&self) -> MlirBlock {
        self.block
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Block::new(vec![]);
    }
}
