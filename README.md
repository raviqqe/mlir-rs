# Melior

[![GitHub Action](https://img.shields.io/github/actions/workflow/status/raviqqe/melior/test.yaml?branch=main&style=flat-square)](https://github.com/raviqqe/melior/actions?query=workflow%3Atest)
[![Crate](https://img.shields.io/crates/v/melior.svg?style=flat-square)](https://crates.io/crates/melior)
[![License](https://img.shields.io/github/license/raviqqe/melior.svg?style=flat-square)](LICENSE)

The rustic MLIR bindings for Rust

This crate is a wrapper of [the MLIR C API](https://mlir.llvm.org/docs/CAPI/).

```rust
use melior::{
    Context,
    dialect::{self, arith, func},
    ir::*,
    utility::register_all_dialects,
};

let registry = dialect::Registry::new();
register_all_dialects(&registry);

let context = Context::new();
context.append_dialect_registry(&registry);
context.load_all_available_dialects();

let location = Location::unknown(&context);
let module = Module::new(location);

let index_type = Type::index(&context);

let function = {
    let block = Block::new(&[(index_type, location), (index_type, location)]);

    let sum = block.append_operation(arith::addi(
        block.argument(0).unwrap().into(),
        block.argument(1).unwrap().into(),
        location
    ));

    block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], location));

    let region = Region::new();
    region.append_block(block);

    func::func(
        &context,
        Attribute::parse(&context, "\"add\"").unwrap(),
        Attribute::parse(&context, "(index, index) -> index").unwrap(),
        region,
        location,
    )
};

module.body().append_operation(function);

assert!(module.as_operation().verify());
```

## Goals

Melior aims to provide a simple, safe, and complete API for MLIR with a reasonably sane ownership model represented by the type system in Rust.

## Install

```sh
cargo add melior
```

### Dependencies

[LLVM/MLIR 16](https://llvm.org/) needs to be installed on your system. On Linux and macOS, you can install it via [Homebrew](https://brew.sh).

```sh
brew install llvm@16
```

## Documentation

On [GitHub Pages](https://raviqqe.github.io/melior/melior/).

## Contribution

Contribution is welcome! But, Melior is still in the alpha stage as well as the MLIR C API. Note that the API is unstable and can have breaking changes in the future.

### Technical notes

- We always use `&T` for MLIR objects instead of `&mut T` to mitigate the intricacy of representing a loose ownership model of the MLIR C API in Rust.
- Only UTF-8 is supported as string encoding.
  - Most string conversion between Rust and C is cached internally.

### Naming conventions

- `Mlir<X>` objects are named `<X>` if they have no destructor. Otherwise, they are named `<X>` for owned objects and `<X>Ref` for borrowed references.
- `mlir<X>Create` functions are renamed as `<X>::new`.
- `mlir<X>Get<Y>` functions are renamed as follows:
  - If the resulting objects refer to `&self`, they are named `<X>::as_<Y>`.
  - Otherwise, they are named just `<X>::<Y>` and may have arguments, such as position indices.

## References

- The raw C binding generation depends on [femtomc/mlir-sys](https://github.com/femtomc/mlir-sys).
- The overall design is inspired by [TheDan64/inkwell](https://github.com/TheDan64/inkwell).

## License

[Apache 2.0](LICENSE)
