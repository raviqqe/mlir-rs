use autophagy_macro::instruction;

#[instruction]
pub fn foo() {}

#[instruction]
pub fn bar(x: usize) -> usize {
    x
}
