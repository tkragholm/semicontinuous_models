use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use faer::Mat;
use semicontinuous_models::models::mtp::sampler::{benchmark_weighted_xtx, benchmark_weighted_xtz};
use semicontinuous_models::models::two_part::benchmark_two_part_weighted_xtz;

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("benchmark size fits u32"))
}

fn build_inputs(n: usize, p: usize) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
    let x = Mat::from_fn(n, p, |row, col| {
        let row_f = usize_to_f64(row);
        let col_f = usize_to_f64(col);
        0.2f64.mul_add((0.0001 * row_f * (col_f + 1.0)).cos(), 0.17f64.mul_add(col_f, 0.0003 * row_f).sin())
    });
    let weights = Mat::from_fn(n, 1, |row, _| {
        let row_f = usize_to_f64(row);
        0.95f64.mul_add((0.001 * row_f).sin().abs() , 0.05)
    });
    let response = Mat::from_fn(n, 1, |row, _| {
        let row_f = usize_to_f64(row);
        0.1f64.mul_add((0.007 * row_f).sin(), (0.002 * row_f).cos())
    });

    (x, weights, response)
}

fn bench_weighted_xtx(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("mtp_weighted_xtx");
    for (n, p) in [(200usize, 5usize), (1_000, 10), (5_000, 20)] {
        let (x, weights, _) = build_inputs(n, p);
        let throughput = u64::try_from(n * p).expect("benchmark shape fits u64");
        group.throughput(Throughput::Elements(throughput));
        group.bench_function(format!("current/n{n}_p{p}"), |bencher| {
            bencher.iter(|| {
                let result = benchmark_weighted_xtx(black_box(&x), black_box(&weights));
                black_box(result);
            });
        });
        group.bench_function(format!("faer_ref/n{n}_p{p}"), |bencher| {
            bencher.iter(|| {
                let weighted_x = Mat::from_fn(n, p, |row, col| {
                    x[(row, col)] * weights[(row, 0)].max(0.0).sqrt()
                });
                let result = weighted_x.transpose() * &weighted_x;
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_weighted_xtz(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("mtp_weighted_xtz");
    for (n, p) in [(200usize, 5usize), (1_000, 10), (5_000, 20)] {
        let (x, weights, response) = build_inputs(n, p);
        let throughput = u64::try_from(n * p).expect("benchmark shape fits u64");
        group.throughput(Throughput::Elements(throughput));
        group.bench_function(format!("weighted_xtz/n{n}_p{p}"), |bencher| {
            bencher.iter(|| {
                let result = benchmark_weighted_xtz(
                    black_box(&x),
                    black_box(&weights),
                    black_box(&response),
                );
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_two_part_weighted_xtz(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("two_part_weighted_xtz");
    for (n, p) in [(200usize, 5usize), (1_000, 10), (5_000, 20)] {
        let (x, weights, response) = build_inputs(n, p);
        let throughput = u64::try_from(n * p).expect("benchmark shape fits u64");
        group.throughput(Throughput::Elements(throughput));
        group.bench_function(format!("weighted_xtz/n{n}_p{p}"), |bencher| {
            bencher.iter(|| {
                let result = benchmark_two_part_weighted_xtz(
                    black_box(&x),
                    black_box(&weights),
                    black_box(&response),
                );
                black_box(result);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_weighted_xtx,
    bench_weighted_xtz,
    bench_two_part_weighted_xtz
);
criterion_main!(benches);
