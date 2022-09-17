//! IR objects and builders.

mod attribute;
mod block;
mod identifier;
mod location;
mod module;
pub mod operation;
mod region;
pub mod r#type;
mod value;

pub use self::{
    attribute::Attribute,
    block::{Block, BlockRef},
    identifier::Identifier,
    location::Location,
    module::Module,
    operation::{Operation, OperationRef},
    r#type::Type,
    region::{Region, RegionRef},
    value::Value,
};
