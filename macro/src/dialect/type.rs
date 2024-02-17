use tblgen::record::Record;

#[derive(Debug, Clone, Copy)]
pub struct Type<'a> {
    optional: bool,
    variadic: bool,
    variadic_of_variadic: bool,
}

impl<'a> Type<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self {
            optional: record.subclass_of("Optional"),
            variadic: record.subclass_of("Variadic"),
            variadic_of_variadic: record.subclass_of("VariadicOfVariadic"),
        }
    }

    pub fn is_optional(&self) -> bool {
        self.optional
    }

    pub fn is_variadic(&self) -> bool {
        self.variadic
    }

    // TODO Support variadic-of-variadic.
    #[allow(unused)]
    pub fn is_variadic_of_variadic(&self) -> bool {
        self.variadic_of_variadic
    }

    pub fn is_unfixed(&self) -> bool {
        self.is_variadic() || self.is_optional()
    }
}
