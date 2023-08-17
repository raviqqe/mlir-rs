use std::io;

use comrak::{arena_tree::NodeEdge, format_commonmark, nodes::NodeValue, parse_document, Arena};
use convert_case::{Case, Casing};
use once_cell::sync::Lazy;
use proc_macro2::Ident;
use quote::format_ident;
use regex::{Captures, Regex};

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

pub fn sanitize_documentation(string: &str) -> Result<String, io::Error> {
    let mut arena = Arena::new();
    let node = parse_document(&mut arena, string, &Default::default());

    for node in node.traverse() {
        match node {
            NodeEdge::Start(node) => {
                let mut ast = node.data.borrow_mut();

                match &mut ast.value {
                    NodeValue::CodeBlock(block) => {
                        if block.info == "" {
                            block.info = "text".into();
                        }
                    }
                    _ => {}
                }
            }
            NodeEdge::End(_) => {}
        }
    }

    let mut string = String::new();

    format_commonmark(node, &Default::default(), &mut string)?;

    Ok(string)
}

static NAME_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(bf_16|f_16|f_32|f_64|i_8|i_16|i_32|i_64|float_8_e_[0-9]_m_[0-9](_fn)?)"#)
        .unwrap()
});

pub fn map_name(name: &str) -> String {
    NAME_PATTERN
        .replace_all(name, |captures: &Captures| {
            captures.get(0).unwrap().as_str().replace('_', "")
        })
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_normal_type_name() {
        assert_eq!(map_name("index"), "index");
        assert_eq!(map_name("integer"), "integer");
    }

    #[test]
    fn map_integer_type_name() {
        assert_eq!(map_name("i_64"), "i64");
    }

    #[test]
    fn map_float_type_name() {
        assert_eq!(map_name("f_64"), "f64");
        assert_eq!(map_name("float_8_e_5_m_2"), "float8e5m2");
        assert_eq!(map_name("float_8_e_4_m_3_fn"), "float8e4m3fn");
    }

    mod sanitize_documentation {
        use super::*;

        #[test]
        fn sanitize_code_block() {
            assert_eq!(
                &sanitize_documentation("```\nfoo\n```"),
                "```text\nfoo\n```"
            );
        }

        #[test]
        fn sanitize_code_blocks() {
            assert_eq!(
                &sanitize_documentation("```\nfoo\n```\n```\nbar\n```"),
                "```text\nfoo\n```\n```text\nbar\n```"
            );
        }
    }
}
