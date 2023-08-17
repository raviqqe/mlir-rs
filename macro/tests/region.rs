mod utility;

use melior::ir::{Block, Location, Region};
use utility::*;

melior_macro::dialect! {
    name: "region_test",
    td_file: "macro/tests/ods_include/region.td",
}

#[test]
fn single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let op = {
        let block = Block::new(&[]);
        let r1 = Region::new();
        r1.append_block(block);
        region_test::single(r1, location)
    };

    assert!(op.default_region().unwrap().first_block().is_some());
}

#[test]
fn variadic_after_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let op = {
        let block = Block::new(&[]);
        let (r1, r2, r3) = (Region::new(), Region::new(), Region::new());
        r2.append_block(block);
        region_test::variadic(r1, vec![r2, r3], location)
    };

    let op2 = {
        let block = Block::new(&[]);
        let (r1, r2, r3) = (Region::new(), Region::new(), Region::new());
        r2.append_block(block);
        region_test::VariadicOp::builder(location)
            .default_region(r1)
            .other_regions(vec![r2, r3])
            .build()
    };

    assert_eq!(op.operation().to_string(), op2.operation().to_string());

    assert!(op.default_region().unwrap().first_block().is_none());
    assert_eq!(op.other_regions().count(), 2);
    assert!(op.other_regions().next().unwrap().first_block().is_some());
    assert!(op.other_regions().nth(1).unwrap().first_block().is_none());
}
