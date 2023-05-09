//! IR objects and builders.

mod affine_map;
mod attribute;
pub mod block;
mod identifier;
mod location;
mod module;
pub mod operation;
mod region;
pub mod r#type;
mod value;

pub use self::{
    affine_map::AffineMap,
    attribute::{Attribute, IntegerAttribute},
    block::{Block, BlockRef},
    identifier::Identifier,
    location::Location,
    module::Module,
    operation::{Operation, OperationRef},
    r#type::{Type, TypeLike},
    region::{Region, RegionRef},
    value::{Value, ValueLike},
};
