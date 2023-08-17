use super::error::Error;
use comrak::{arena_tree::NodeEdge, format_commonmark, nodes::NodeValue, parse_document, Arena};

pub fn sanitize_documentation(string: &str) -> Result<String, Error> {
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

    let mut buffer = vec![];

    format_commonmark(node, &Default::default(), &mut buffer)?;

    Ok(String::from_utf8(buffer)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn sanitize_code_block() {
        assert_eq!(
            &sanitize_documentation("```\nfoo\n```\n").unwrap(),
            "``` text\nfoo\n```\n"
        );
    }

    #[test]
    fn sanitize_code_blocks() {
        assert_eq!(
            &sanitize_documentation("```\nfoo\n```\n\n```\nbar\n```\n").unwrap(),
            "``` text\nfoo\n```\n\n``` text\nbar\n```\n"
        );
    }
}
