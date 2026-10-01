//! Fits must be bit-identical across repeated calls and across thread counts.
//!
//! The robust covariance of a poorly identified model amplifies any change in
//! summation order, so a reduction whose order depends on thread scheduling or
//! on hash-map iteration order shows up as standard errors that differ between
//! runs. These tests fit each model several times, in the calling thread and in
//! rayon pools of 1, 2 and 8 threads, and compare every coefficient and every
//! standard error bit for bit.

use faer::Mat;
use semicontinuous_models::{
    FitOptions, LogNormalOptions, ModelInput, TweedieOptions, fit_lognormal_smearing_input,
    fit_tweedie_input, fit_two_part_input,
};

const N_ROWS: usize = 12_000;
const N_COLS: usize = 8;
const REPEATS: usize = 2;
const THREAD_COUNTS: [usize; 3] = [1, 2, 8];

/// SplitMix64, so the fixture does not depend on an RNG crate's stream.
struct Stream(u64);

impl Stream {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// A cost-shaped fixture: intercept, a binary exposure, continuous covariates,
/// about 40% zeros, right-skewed positives, positive weights, and family-sized
/// clusters of one to four rows.
fn fixture(weighted: bool) -> ModelInput {
    let mut stream = Stream(20_261_001);
    let mut x = Mat::<f64>::zeros(N_ROWS, N_COLS);
    for i in 0..N_ROWS {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = if stream.uniform() < 0.3 { 1.0 } else { 0.0 };
        for j in 2..N_COLS {
            x[(i, j)] = 2.0f64.mul_add(stream.uniform(), -1.0);
        }
    }
    let mut y = Mat::<f64>::zeros(N_ROWS, 1);
    for i in 0..N_ROWS {
        let eta = 0.4f64.mul_add(x[(i, 1)], 0.2 * x[(i, 2)] - 0.1 * x[(i, 3)]);
        if stream.uniform() < 0.4 {
            continue;
        }
        let noise = -stream.uniform().max(1e-12).ln();
        y[(i, 0)] = 1_000.0 * (8.0f64 + eta).exp() / 3_000.0 * noise;
    }
    let mut clusters = Vec::with_capacity(N_ROWS);
    let mut cluster = 0_u64;
    while clusters.len() < N_ROWS {
        let size = 1 + (stream.next_u64() % 4) as usize;
        for _ in 0..size.min(N_ROWS - clusters.len()) {
            clusters.push(cluster * 7_919 + 13);
        }
        cluster += 1;
    }
    let input = ModelInput::new(x, y).with_cluster_ids(clusters);
    if weighted {
        let w = Mat::from_fn(N_ROWS, 1, |_, _| 0.5 + stream.uniform());
        input.with_sample_weights(w)
    } else {
        input
    }
}

fn bits(values: &[&Mat<f64>]) -> Vec<u64> {
    values
        .iter()
        .flat_map(|m| (0..m.nrows()).map(move |i| m[(i, 0)].to_bits()))
        .collect()
}

fn two_part_bits(input: &ModelInput) -> Vec<u64> {
    let options = FitOptions::builder().robust_se(true).build();
    let (model, report) = fit_two_part_input(input, options).expect("two-part fit");
    bits(&[
        &model.beta_logit,
        &model.beta_gamma,
        report.se_logit.as_ref().expect("logit SE"),
        report.se_gamma.as_ref().expect("gamma SE"),
    ])
}

fn tweedie_bits(input: &ModelInput, power: f64) -> Vec<u64> {
    let options = TweedieOptions::builder().robust_se(true).build();
    let (model, report) = fit_tweedie_input(input, power, options).expect("tweedie fit");
    bits(&[&model.beta, report.se.as_ref().expect("tweedie SE")])
}

fn lognormal_bits(input: &ModelInput) -> Vec<u64> {
    let options = LogNormalOptions::builder().robust_se(true).build();
    let (model, report) = fit_lognormal_smearing_input(input, options).expect("lognormal fit");
    bits(&[&model.beta, report.se.as_ref().expect("lognormal SE")])
}

fn assert_reproducible(label: &str, fit: impl Fn() -> Vec<u64> + Sync) {
    let reference = fit();
    for repeat in 0..REPEATS {
        assert_eq!(fit(), reference, "{label}: repeat {repeat} differs");
    }
    for threads in THREAD_COUNTS {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool");
        assert_eq!(
            pool.install(&fit),
            reference,
            "{label}: {threads} threads differs"
        );
    }
}

fn without_clusters(input: &ModelInput) -> ModelInput {
    let mut input = input.clone();
    input.cluster_ids = None;
    input
}

#[test]
fn two_part_fits_are_bit_identical() {
    for weighted in [false, true] {
        let clustered = fixture(weighted);
        let plain = without_clusters(&clustered);
        assert_reproducible("two-part clustered", || two_part_bits(&clustered));
        assert_reproducible("two-part", || two_part_bits(&plain));
    }
}

#[test]
fn tweedie_fits_are_bit_identical() {
    for weighted in [false, true] {
        let clustered = fixture(weighted);
        let plain = without_clusters(&clustered);
        for power in [1.5, 2.0] {
            assert_reproducible("tweedie clustered", || tweedie_bits(&clustered, power));
            assert_reproducible("tweedie", || tweedie_bits(&plain, power));
        }
    }
}

#[test]
fn lognormal_fits_are_bit_identical() {
    let clustered = fixture(false);
    let plain = without_clusters(&clustered);
    assert_reproducible("lognormal clustered", || lognormal_bits(&clustered));
    assert_reproducible("lognormal", || lognormal_bits(&plain));
}
