# Melior

The rustic MLIR bindings for Rust

This crate is a wrapper of [the MLIR C API](https://mlir.llvm.org/docs/CAPI/).

## Dependencies

[LLVM/MLIR 15](https://llvm.org/) needs to be installed on your system. On Linux and macOS, you can install it via [Homebrew](https://brew.sh).

```sh
brew install llvm@15
```

## Contribution

Contribution is welcome! But, Melior is still in the alpha stage as well as the MLIR C API. Note that the API is unstable and can have breaking changes in the future.

### Design decisions

- Only UTF-8 is supported as string encoding.
  - Most string conversion between Rust and C is cached internally.

### Naming conventions

- `mlir<X>Create*` functions are renamed as `<X>::new`.
- `mlir<X>Get<Y>` functions are renamed as follows:
  - If the resulting objects refer to `&self`, they are named `<X>::as_<Y>`.
  - Otherwise, they are named just `<X>::<Y>` and may have arguments, such as position indices.

## References

- The raw C binding generation depends on [femtomc/mlir-sys](https://github.com/femtomc/mlir-sys).
- The overall design is inspired by [TheDan64/inkwell](https://github.com/TheDan64/inkwell).

## License

[Apache 2.0](LICENSE)
