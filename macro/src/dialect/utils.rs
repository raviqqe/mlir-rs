use convert_case::{Case, Casing};
use proc_macro2::Ident;
use quote::format_ident;

static RESERVED_NAMES: &[&str] = &["name", "operation", "builder"];

pub fn sanitize_name_snake(name: &str) -> Ident {
    sanitize_name(&name.to_case(Case::Snake))
}

pub fn sanitize_name(name: &str) -> Ident {
    // Replace any "." with "_"
    let mut name = name.replace('.', "_");

    // Add "_" suffix to avoid conflicts with existing methods
    if RESERVED_NAMES.contains(&name.as_str())
        || name
            .chars()
            .next()
            .expect("name has at least one char")
            .is_numeric()
    {
        name = format!("_{}", name);
    }

    // Try to parse the string as an ident, and prefix the identifier
    // with "r#" if it is not a valid identifier.
    syn::parse_str::<Ident>(&name).unwrap_or(format_ident!("r#{}", name))
}
