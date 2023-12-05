#[derive(Clone, Debug)]
pub enum VariadicKind {
    Simple {
        variable_length_seen: bool,
    },
    SameSize {
        variable_length_count: usize,
        preceding_simple_count: usize,
        preceding_variadic_count: usize,
    },
    AttributeSized,
}

impl VariadicKind {
    pub fn new(variable_length_count: usize, same_size: bool, attribute_sized: bool) -> Self {
        if variable_length_count <= 1 {
            VariadicKind::Simple {
                variable_length_seen: false,
            }
        } else if same_size {
            VariadicKind::SameSize {
                variable_length_count,
                preceding_simple_count: 0,
                preceding_variadic_count: 0,
            }
        } else if attribute_sized {
            VariadicKind::AttributeSized {}
        } else {
            unimplemented!()
        }
    }
}
