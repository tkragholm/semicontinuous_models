use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use semicontinuous_models::models::mtp::sampler::benchmark_sample_standard_normals;

fn bench_sample_standard_normals(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("mtp_sample_standard_normals");
    for count in [100_000usize, 1_000_000] {
        group.throughput(Throughput::Elements(
            u64::try_from(count).expect("count fits u64"),
        ));
        group.bench_function(format!("count={count}"), |bencher| {
            bencher.iter(|| {
                let sum = benchmark_sample_standard_normals(black_box(count), 17_431);
                black_box(sum);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sample_standard_normals);
criterion_main!(benches);
