use proc_macro2::{Ident, TokenStream};
use syn::Type;

pub trait OperationField {
    fn name(&self) -> &str;
    fn singular_identifier(&self) -> &Ident;
    fn plural_kind_identifier(&self) -> Ident;
    fn parameter_type(&self) -> Type;
    fn return_type(&self) -> Type;
    fn is_optional(&self) -> bool;
    fn add_arguments(&self, name: &Ident) -> TokenStream;

    // TODO Remove this.
    fn is_result(&self) -> bool;
}
