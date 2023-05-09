use convert_case::{Case, Casing};
use once_cell::sync::Lazy;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use regex::{Captures, Regex};
use std::error::Error;

static FLOAT_8_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"float_8_e_[0-9]_m_[0-9](_fn)?"#).unwrap());

pub fn map_name(name: &str) -> String {
    match name {
        "bf_16" | "f_16" | "f_32" | "f_64" | "i_8" | "i_16" | "i_32" | "i_64" => {
            name.replace('_', "")
        }
        name => FLOAT_8_PATTERN
            .replace(name, |captures: &Captures| {
                captures.get(0).unwrap().as_str().replace('_', "")
            })
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_normal_type_name() {
        assert_eq!(map_type_name("index"), "index");
        assert_eq!(map_type_name("integer"), "integer");
    }

    #[test]
    fn map_float_type_name() {
        assert_eq!(map_type_name("float_8_e_5_m_2"), "float8e5m2");
        assert_eq!(map_type_name("float_8_e_4_m_3_fn"), "float8e4m3fn");
    }
}
