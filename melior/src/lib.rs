//! Melior is the rustic MLIR bindings for Rust. It aims to provide a simple,
//! safe, and complete API for MLIR with a reasonably sane ownership model
//! represented by the type system in Rust.
//!
//! This crate is a wrapper of [the MLIR C API](https://mlir.llvm.org/docs/CAPI/).
//!
//! # Dependencies
//!
//! [LLVM/MLIR 16](https://llvm.org/) needs to be installed on your system. On Linux and macOS, you can install it via [Homebrew](https://brew.sh).
//!
//! ```sh
//! brew install llvm@16
//! ```
//!
//! # Safety
//!
//! Although Melior aims to be completely safe, some part of the current API is
//! not.
//!
//! - Access to operations, types, or attributes that belong to dialects not
//!   loaded in contexts can lead to runtime errors or segmentation faults in
//!   the worst case.
//!   - Fix plan: Load all dialects by default on creation of contexts, and
//!     provide unsafe constructors of contexts for advanced users.
//! - IR object references returned from functions that move ownership of
//!   arguments might get invalidated later.
//!   - This is because we need to borrow `&self` rather than `&mut self` to
//!     return such references.
//!   - e.g. `Region::append_block()`
//!   - Fix plan: Use dynamic check, such as `RefCell`, for the objects.
//!
//! # Examples
//!
//! ## Building a function to add integers
//!
//! ```rust
//! use melior::{
//!     Context,
//!     dialect::{self, arith, func},
//!     ir::*,
//!     utility::register_all_dialects,
//! };
//!
//! let registry = dialect::Registry::new();
//! register_all_dialects(&registry);
//!
//! let context = Context::new();
//! context.append_dialect_registry(&registry);
//! context.load_all_available_dialects();
//!
//! let location = Location::unknown(&context);
//! let module = Module::new(location);
//!
//! let index_type = Type::index(&context);
//!
//! let function = {
//!     let block = Block::new(&[(index_type, location), (index_type, location)]);
//!
//!     let sum = block.append_operation(arith::addi(
//!         block.argument(0).unwrap().into(),
//!         block.argument(1).unwrap().into(),
//!         location
//!     ));
//!
//!     block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], location));
//!
//!     let region = Region::new();
//!     region.append_block(block);
//!
//!     func::func(
//!         &context,
//!         Attribute::parse(&context, "\"add\"").unwrap(),
//!         Attribute::parse(&context, "(index, index) -> index").unwrap(),  
//!         region,
//!         location,
//!     )
//! };
//!
//! module.body().append_operation(function);
//!
//! assert!(module.as_operation().verify());
//! ```

mod context;
pub mod dialect;
mod error;
mod execution_engine;
pub mod ir;
mod logical_result;
pub mod pass;
mod string_ref;
#[cfg(test)]
mod test;
pub mod utility;

pub use self::{
    context::{Context, ContextRef},
    error::Error,
    execution_engine::ExecutionEngine,
    string_ref::StringRef,
};

#[cfg(test)]
mod tests {
    use crate::{
        context::Context,
        dialect,
        ir::{operation, r#type, Attribute, Block, Identifier, Location, Module, Region, Type},
        test::load_all_dialects,
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
        let registry = dialect::Registry::new();
        let context = Context::new();
        context.append_dialect_registry(&registry);
        let module = Module::new(Location::unknown(&context));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn build_add() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let integer_type = r#type::Integer::new(&context, 64).into();

        let function = {
            let block = Block::new(&[(integer_type, location), (integer_type, location)]);

            let sum = block.append_operation(
                operation::Builder::new("arith.addi", location)
                    .add_operands(&[
                        block.argument(0).unwrap().into(),
                        block.argument(1).unwrap().into(),
                    ])
                    .add_results(&[integer_type])
                    .build(),
            );

            block.append_operation(
                operation::Builder::new("func.return", Location::unknown(&context))
                    .add_operands(&[sum.result(0).unwrap().into()])
                    .build(),
            );

            let region = Region::new();
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
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let memref_type = Type::parse(&context, "memref<?xf32>").unwrap();

        let function = {
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
                        function_block.argument(0).unwrap().into(),
                        zero.result(0).unwrap().into(),
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
                            function_block.argument(0).unwrap().into(),
                            loop_block.argument(0).unwrap().into(),
                        ])
                        .add_results(&[f32_type])
                        .build(),
                );

                let rhs = loop_block.append_operation(
                    operation::Builder::new("memref.load", location)
                        .add_operands(&[
                            function_block.argument(1).unwrap().into(),
                            loop_block.argument(0).unwrap().into(),
                        ])
                        .add_results(&[f32_type])
                        .build(),
                );

                let add = loop_block.append_operation(
                    operation::Builder::new("arith.addf", location)
                        .add_operands(&[
                            lhs.result(0).unwrap().into(),
                            rhs.result(0).unwrap().into(),
                        ])
                        .add_results(&[f32_type])
                        .build(),
                );

                loop_block.append_operation(
                    operation::Builder::new("memref.store", location)
                        .add_operands(&[
                            add.result(0).unwrap().into(),
                            function_block.argument(0).unwrap().into(),
                            loop_block.argument(0).unwrap().into(),
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
                            zero.result(0).unwrap().into(),
                            dim.result(0).unwrap().into(),
                            one.result(0).unwrap().into(),
                        ])
                        .add_regions(vec![loop_region])
                }
                .build(),
            );

            function_block.append_operation(
                operation::Builder::new("func.return", Location::unknown(&context)).build(),
            );

            let function_region = Region::new();
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
