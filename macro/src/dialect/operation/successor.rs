use super::SequenceInfo;
use crate::dialect::types::SuccessorConstraint;

pub struct Successor {
    constraint: SuccessorConstraint<'a>,
    sequence_info: SequenceInfo,
}

impl Successor {
    pub fn new(constraint: SuccessorConstraint<'a>, sequence_info: SequenceInfo) -> Self {
        Self {
            constraint,
            sequence_info,
        }
    }
}
