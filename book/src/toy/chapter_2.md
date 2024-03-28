# Emitting Basic MLIR

Simplest dialect, that doesnt add any operations:
```tablegen
// This include is necessary to use `Dialect`
include "mlir/IR/OpBase.td"

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  let name = "toy";
}
```

Generate code for dialect:
```rust
// Generate rust code for the toy dialect.
melior::dialect! {
    name: "toy", // Name of the dialect. Should match `name` from tablegen file
    td_file: "examples/toy/src/include/toy.td", // Path starting from crate root (where `Cargo.toml` is located)
}
```
This will expand to:
```rust
/**`toy` dialect.*/
pub mod toy {}
```

Load dialects:
```rust
let registry = DialectRegistry::new();
register_all_dialects(&registry);

let context = Context::new();
context.append_dialect_registry(&registry);
context.load_all_available_dialects();
context.set_allow_unregistered_dialects(true);
```