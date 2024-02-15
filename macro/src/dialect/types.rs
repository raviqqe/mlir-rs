use tblgen::record::Record;

#[derive(Debug, Clone, Copy)]
pub struct RegionConstraint<'a>(Record<'a>);

impl<'a> RegionConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    pub fn is_variadic(&self) -> bool {
        self.0.subclass_of("VariadicRegion")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SuccessorConstraint<'a>(Record<'a>);

impl<'a> SuccessorConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    pub fn is_variadic(&self) -> bool {
        self.0.subclass_of("VariadicSuccessor")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TypeConstraint<'a>(Record<'a>);

impl<'a> TypeConstraint<'a> {
    pub fn new(record: Record<'a>) -> Self {
        Self(record)
    }

    pub fn is_optional(&self) -> bool {
        self.0.subclass_of("Optional")
    }

    pub fn is_variadic(&self) -> bool {
        self.0.subclass_of("Variadic")
    }

    // TODO Support variadic-of-variadic.
    #[allow(unused)]
    pub fn is_variadic_of_variadic(&self) -> bool {
        self.0.subclass_of("VariadicOfVariadic")
    }

    pub fn is_unfixed(&self) -> bool {
        self.is_variadic() || self.is_optional()
    }
}
