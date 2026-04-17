use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use faer::Mat;
use semicontinuous_models::models::two_part::{ConfidenceInterval, bootstrap_percentile_ci};

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("benchmark size fits u32"))
}

fn bootstrap_percentile_ci_sorted_reference(
    betas: &[Mat<f64>],
    alpha: f64,
) -> Vec<ConfidenceInterval> {
    if betas.is_empty() {
        return Vec::new();
    }
    let n = betas[0].nrows();
    let (lower_idx, upper_idx) =
        semicontinuous_models::utils::boot_index_bounds(alpha, betas.len());
    let mut intervals = Vec::with_capacity(n);
    for col in 0..n {
        let mut values = betas.iter().map(|b| b[(col, 0)]).collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        let lower = values[lower_idx.min(values.len().saturating_sub(1))];
        let upper = values[upper_idx.min(values.len().saturating_sub(1))];
        intervals.push(ConfidenceInterval { lower, upper });
    }
    intervals
}

fn build_betas(reps: usize, coefficients: usize) -> Vec<Mat<f64>> {
    (0..reps)
        .map(|draw| {
            Mat::from_fn(coefficients, 1, |coef, _| {
                let d = usize_to_f64(draw);
                let c = usize_to_f64(coef);
                0.001f64.mul_add(-d, 0.05f64.mul_add(c, (0.17f64.mul_add(d, 0.31 * c)).sin()))
            })
        })
        .collect()
}

fn bench_two_part_bootstrap_percentile_ci(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("two_part_bootstrap_percentile_ci");
    let coefficients = 12usize;
    for reps in [200usize, 999, 5_000] {
        let betas = build_betas(reps, coefficients);
        group.throughput(Throughput::Elements((reps as u64) * (coefficients as u64)));
        group.bench_function(format!("current_select/reps={reps}"), |bench| {
            bench.iter(|| {
                let _ = bootstrap_percentile_ci(&betas, 0.05);
            });
        });
        group.bench_function(format!("reference_sort/reps={reps}"), |bench| {
            bench.iter(|| {
                let _ = bootstrap_percentile_ci_sorted_reference(&betas, 0.05);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_two_part_bootstrap_percentile_ci);
criterion_main!(benches);
