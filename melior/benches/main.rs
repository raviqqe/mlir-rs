use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use melior::{Context, StringRef};

const ITERATION_COUNT: usize = 1000000;

fn generate_strings() -> Vec<String> {
    (0..ITERATION_COUNT)
        .map(|number| number.to_string())
        .collect()
}

fn string_ref_create(bencher: &mut Bencher) {
    let context = Context::new();
    let strings = generate_strings();

    bencher.iter(|| {
        for string in &strings {
            let _ = StringRef::new( string.as_str());
        }
    });
}

fn string_ref_create_cached(bencher: &mut Bencher) {
    let context = Context::new();

    bencher.iter(|| {
        for _ in 0..ITERATION_COUNT {
            let _ = StringRef::new( "foo");
        }
    });
}

fn benchmark(criterion: &mut Criterion) {
    criterion.bench_function("string ref create", string_ref_create);
    criterion.bench_function("string ref create cached", string_ref_create_cached);
}

criterion_group!(benchmark_group, benchmark);
criterion_main!(benchmark_group);
