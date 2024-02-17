use super::{OperationFieldLike, VariadicKind};

pub trait OperationElement: OperationFieldLike {
    fn is_variadic(&self) -> bool;
    fn variadic_kind(&self) -> &VariadicKind;

    fn is_unfixed(&self) -> bool {
        self.is_variadic() || self.is_optional()
    }
}
