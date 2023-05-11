use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use melior::StringRef;

fn generate_strings() -> Vec<String> {
    (0..100000).map(|number| number.to_string()).collect()
}

fn string_ref_create(bencher: &mut Bencher) {
    let strings = generate_strings();

    bencher.iter(|| {
        for string in &strings {
            let _ = StringRef::from(string.as_str());
        }
    });
}

fn benchmark(criterion: &mut Criterion) {
    criterion.bench_function("string ref create", string_ref_create);
}

criterion_group!(benchmark_group, benchmark);
criterion_main!(benchmark_group);
