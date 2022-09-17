//! Melior is the rustic MLIR bindings for Rust. It aims to provide a simple,
//! safe, and complete API for MLIR with a reasonably sane ownership model
//! represented by the type system in Rust.
//!
//! This crate is a wrapper of [the MLIR C API](https://mlir.llvm.org/docs/CAPI/).
//!
//! # Dependencies
//!
//! [LLVM/MLIR 15](https://llvm.org/) needs to be installed on your system. On Linux and macOS, you can install it via [Homebrew](https://brew.sh).
//!
//! ```sh
//! brew install llvm@15
//! ```
//!
//! # Examples
//!
//! ## Building a function to add integers
//!
//! ```rust
//! use melior::{
//!     attribute::Attribute,
//!     block::Block,
//!     context::Context,
//!     dialect_registry::DialectRegistry,
//!     identifier::Identifier,
//!     location::Location,
//!     module::Module,
//!     operation::{self, Operation},
//!     region::Region,
//!     r#type::Type,
//!     utility::register_all_dialects,
//! };
//!
//! let registry = DialectRegistry::new();
//! register_all_dialects(&registry);
//!
//! let context = Context::new();
//! context.append_dialect_registry(&registry);
//! context.get_or_load_dialect("func");
//!
//! let location = Location::unknown(&context);
//! let module = Module::new(location);
//!
//! let integer_type = Type::integer(&context, 64);
//!
//! let function = {
//!     let region = Region::new();
//!     let block = Block::new(&[(integer_type, location), (integer_type, location)]);
//!
//!     let sum = block.append_operation(operation::Builder::new("arith.addi", location)
//!             .add_operands(&[*block.argument(0).unwrap(), *block.argument(1).unwrap()])
//!             .add_results(&[integer_type]).build());
//!
//!     block.append_operation(operation::Builder::new("func.return", Location::unknown(&context))
//!             .add_operands(&[*sum.result(0).unwrap()]).build());
//!
//!     region.append_block(block);
//!
//!     operation::Builder::new("func.func", Location::unknown(&context))
//!             .add_attributes(&[
//!                 (
//!                     Identifier::new(&context, "function_type"),
//!                     Attribute::parse(&context, "(i64, i64) -> i64").unwrap(),
//!                 ),
//!                 (
//!                     Identifier::new(&context, "sym_name"),
//!                     Attribute::parse(&context, "\"add\"").unwrap(),
//!                 ),
//!             ])
//!             .add_regions(vec![region]).build()
//! };
//!
//! module.body().append_operation(function);
//!
//! assert!(module.as_operation().verify());
//! ```

pub mod attribute;
pub mod block;
pub mod context;
pub mod dialect;
pub mod dialect_handle;
pub mod dialect_registry;
pub mod error;
pub mod execution_engine;
pub mod identifier;
pub mod location;
pub mod logical_result;
pub mod module;
pub mod operation;
pub mod operation_pass_manager;
pub mod pass;
pub mod pass_manager;
pub mod region;
pub mod string_ref;
pub mod r#type;
pub mod utility;
pub mod value;

#[cfg(test)]
mod tests {
    use crate::{
        attribute::Attribute,
        block::Block,
        context::Context,
        dialect_registry::DialectRegistry,
        identifier::Identifier,
        location::Location,
        module::Module,
        operation::{self},
        r#type::Type,
        region::Region,
        utility::register_all_dialects,
    };

    #[test]
    fn build_module() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn build_module_with_dialect() {
        let registry = DialectRegistry::new();
        let context = Context::new();
        context.append_dialect_registry(&registry);
        let module = Module::new(Location::unknown(&context));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn build_add() {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("func");

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let integer_type = Type::integer(&context, 64);

        let function = {
            let region = Region::new();
            let block = Block::new(&[(integer_type, location), (integer_type, location)]);

            let sum = block.append_operation(
                operation::Builder::new("arith.addi", location)
                    .add_operands(&[*block.argument(0).unwrap(), *block.argument(1).unwrap()])
                    .add_results(&[integer_type])
                    .build(),
            );

            block.append_operation(
                operation::Builder::new("func.return", Location::unknown(&context))
                    .add_operands(&[*sum.result(0).unwrap()])
                    .build(),
            );

            region.append_block(block);

            operation::Builder::new("func.func", Location::unknown(&context))
                .add_attributes(&[
                    (
                        Identifier::new(&context, "function_type"),
                        Attribute::parse(&context, "(i64, i64) -> i64").unwrap(),
                    ),
                    (
                        Identifier::new(&context, "sym_name"),
                        Attribute::parse(&context, "\"add\"").unwrap(),
                    ),
                ])
                .add_regions(vec![region])
                .build()
        };

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn build_sum() {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("func");
        context.get_or_load_dialect("memref");
        context.get_or_load_dialect("scf");

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let memref_type = Type::parse(&context, "memref<?xf32>").unwrap();

        let function = {
            let function_region = Region::new();
            let function_block = Block::new(&[(memref_type, location), (memref_type, location)]);
            let index_type = Type::parse(&context, "index").unwrap();

            let zero = function_block.append_operation(
                operation::Builder::new("arith.constant", location)
                    .add_results(&[index_type])
                    .add_attributes(&[(
                        Identifier::new(&context, "value"),
                        Attribute::parse(&context, "0 : index").unwrap(),
                    )])
                    .build(),
            );

            let dim = function_block.append_operation(
                operation::Builder::new("memref.dim", location)
                    .add_operands(&[
                        *function_block.argument(0).unwrap(),
                        *zero.result(0).unwrap(),
                    ])
                    .add_results(&[index_type])
                    .build(),
            );

            let loop_block = Block::new(&[]);
            loop_block.add_argument(index_type, location);

            let one = function_block.append_operation(
                operation::Builder::new("arith.constant", location)
                    .add_results(&[index_type])
                    .add_attributes(&[(
                        Identifier::new(&context, "value"),
                        Attribute::parse(&context, "1 : index").unwrap(),
                    )])
                    .build(),
            );

            {
                let f32_type = Type::parse(&context, "f32").unwrap();

                let lhs = loop_block.append_operation(
                    operation::Builder::new("memref.load", location)
                        .add_operands(&[
                            *function_block.argument(0).unwrap(),
                            *loop_block.argument(0).unwrap(),
                        ])
                        .add_results(&[f32_type])
                        .build(),
                );

                let rhs = loop_block.append_operation(
                    operation::Builder::new("memref.load", location)
                        .add_operands(&[
                            *function_block.argument(1).unwrap(),
                            *loop_block.argument(0).unwrap(),
                        ])
                        .add_results(&[f32_type])
                        .build(),
                );

                let add = loop_block.append_operation(
                    operation::Builder::new("arith.addf", location)
                        .add_operands(&[*lhs.result(0).unwrap(), *rhs.result(0).unwrap()])
                        .add_results(&[f32_type])
                        .build(),
                );

                loop_block.append_operation(
                    operation::Builder::new("memref.store", location)
                        .add_operands(&[
                            *add.result(0).unwrap(),
                            *function_block.argument(0).unwrap(),
                            *loop_block.argument(0).unwrap(),
                        ])
                        .build(),
                );

                loop_block.append_operation(operation::Builder::new("scf.yield", location).build());
            }

            function_block.append_operation(
                {
                    let loop_region = Region::new();

                    loop_region.append_block(loop_block);

                    operation::Builder::new("scf.for", location)
                        .add_operands(&[
                            *zero.result(0).unwrap(),
                            *dim.result(0).unwrap(),
                            *one.result(0).unwrap(),
                        ])
                        .add_regions(vec![loop_region])
                }
                .build(),
            );

            function_block.append_operation(
                operation::Builder::new("func.return", Location::unknown(&context)).build(),
            );

            function_region.append_block(function_block);

            operation::Builder::new("func.func", Location::unknown(&context))
                .add_attributes(&[
                    (
                        Identifier::new(&context, "function_type"),
                        Attribute::parse(&context, "(memref<?xf32>, memref<?xf32>) -> ()").unwrap(),
                    ),
                    (
                        Identifier::new(&context, "sym_name"),
                        Attribute::parse(&context, "\"sum\"").unwrap(),
                    ),
                ])
                .add_regions(vec![function_region])
                .build()
        };

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
}
