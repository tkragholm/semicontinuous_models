//! Robust standard errors against references that do not share this crate's code.
//!
//! In a correctly specified model with unit dispersion the sandwich and the
//! model-based covariance estimate the same thing, so on a large sample the two
//! standard errors must be close. The fixed fixture below is also checked against
//! HC0 standard errors computed in R from `glm` fits on the same rows, and a
//! clustered fit with one row per cluster must reproduce the unclustered sandwich.

use faer::prelude::*;
use faer::{Mat, Side};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma};
use semicontinuous_models::{
    FitOptions, ModelInput, TweedieOptions, fit_tweedie_input, fit_two_part_input,
};

const N_ROWS: usize = 20_000;

/// SplitMix64, so the fixture does not depend on an RNG crate's stream and the R
/// references stay valid.
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

/// Intercept, a uniform covariate and a binary one. `Pr(y > 0)` follows the
/// logit `0.2 + 0.8 x1 - 0.5 x2`, and a positive `y` is exponential with mean
/// `exp(1 + 0.3 x1 + 0.2 x2)`, so both parts are correctly specified with unit
/// dispersion.
fn two_part_fixture() -> (Mat<f64>, Mat<f64>) {
    let mut stream = Stream(20_261_002);
    let mut x = Mat::<f64>::zeros(N_ROWS, 3);
    let mut y = Mat::<f64>::zeros(N_ROWS, 1);
    for i in 0..N_ROWS {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = 2.0f64.mul_add(stream.uniform(), -1.0);
        x[(i, 2)] = if stream.uniform() < 0.3 { 1.0 } else { 0.0 };
        let eta_logit = 0.8f64.mul_add(x[(i, 1)], 0.2) - 0.5 * x[(i, 2)];
        let prob = 1.0 / (1.0 + (-eta_logit).exp());
        let positive = stream.uniform() < prob;
        let draw = -(1.0 - stream.uniform()).ln();
        if positive {
            let mean = 0.3f64.mul_add(x[(i, 1)], 1.0 + 0.2 * x[(i, 2)]).exp();
            y[(i, 0)] = mean * draw;
        }
    }
    (x, y)
}

fn positive_rows(x: &Mat<f64>, y: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    let rows: Vec<usize> = (0..y.nrows()).filter(|&i| y[(i, 0)] > 0.0).collect();
    (
        Mat::from_fn(rows.len(), x.ncols(), |i, j| x[(rows[i], j)]),
        Mat::from_fn(rows.len(), 1, |i, _| y[(rows[i], 0)]),
    )
}

/// Model-based standard errors of a log-link Tweedie fit with unit dispersion,
/// `sqrt(diag((X' diag(mu^(2 - power)) X)^-1))`. The crate reports no model-based
/// covariance for Tweedie fits, so the test forms it here.
fn tweedie_model_based_se(x: &Mat<f64>, beta: &Mat<f64>, power: f64) -> Vec<f64> {
    let p = x.ncols();
    let mut information = Mat::<f64>::zeros(p, p);
    for i in 0..x.nrows() {
        let eta: f64 = (0..p).map(|j| x[(i, j)] * beta[(j, 0)]).sum();
        let weight = eta.exp().powf(2.0 - power);
        for j in 0..p {
            for k in 0..p {
                information[(j, k)] += weight * x[(i, j)] * x[(i, k)];
            }
        }
    }
    let inverse = information
        .llt(Side::Lower)
        .expect("positive-definite information")
        .solve(Mat::<f64>::identity(p, p));
    (0..p).map(|j| inverse[(j, j)].sqrt()).collect()
}

fn column(m: &Mat<f64>) -> Vec<f64> {
    (0..m.nrows()).map(|i| m[(i, 0)]).collect()
}

fn assert_close(label: &str, got: &[f64], want: &[f64], rel_tol: f64) {
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            ((g - w) / w).abs() < rel_tol,
            "{label}[{k}]: got {g}, want {w} (tolerance {rel_tol})"
        );
    }
}

/// HC0 standard errors from R 4 on the rows of `two_part_fixture`:
/// `glm(pos ~ x1 + x2, binomial)` for the logit part and
/// `glm(y ~ x1 + x2, Gamma(link = "log"), subset = y > 0)` for the positive part,
/// each with `B %*% crossprod(X * r) %*% B`, `B = solve(crossprod(X * sqrt(w)))`,
/// `w` the working weights and `r` the score residual.
const R_HC0_LOGIT: [f64; 3] = [0.017_497_413_181, 0.025_828_785_236, 0.031_797_081_066];
const R_HC0_GAMMA: [f64; 3] = [0.011_792_762_166, 0.018_088_586_845, 0.022_420_902_035];

#[test]
fn two_part_logit_robust_se_matches_model_based_and_r() {
    let (x, y) = two_part_fixture();
    let input = ModelInput::new(x, y);
    let (_, model_based) = fit_two_part_input(&input, FitOptions::default()).expect("fit");
    let (_, robust) =
        fit_two_part_input(&input, FitOptions::builder().robust_se(true).build()).expect("fit");

    let model_se = column(model_based.se_logit.as_ref().expect("model-based SE"));
    let robust_se = column(robust.se_logit.as_ref().expect("robust SE"));
    assert_close("logit robust vs model-based", &robust_se, &model_se, 0.1);
    assert_close("logit robust vs R", &robust_se, &R_HC0_LOGIT, 1e-6);
    assert_close(
        "gamma robust vs R",
        &column(robust.se_gamma.as_ref().expect("robust SE")),
        &R_HC0_GAMMA,
        1e-6,
    );
}

#[test]
fn two_part_singleton_clusters_reproduce_the_unclustered_sandwich() {
    let (x, y) = two_part_fixture();
    let clusters: Vec<u64> = (0..N_ROWS as u64).collect();
    let options = FitOptions::builder().robust_se(true).build();
    let (_, plain) = fit_two_part_input(&ModelInput::new(x.clone(), y.clone()), options)
        .expect("unclustered fit");
    let (_, clustered) =
        fit_two_part_input(&ModelInput::new(x, y).with_cluster_ids(clusters), options)
            .expect("clustered fit");
    assert_close(
        "logit",
        &column(clustered.se_logit.as_ref().expect("SE")),
        &column(plain.se_logit.as_ref().expect("SE")),
        1e-9,
    );
    assert_close(
        "gamma",
        &column(clustered.se_gamma.as_ref().expect("SE")),
        &column(plain.se_gamma.as_ref().expect("SE")),
        1e-9,
    );
}

#[test]
fn tweedie_gamma_robust_se_matches_model_based_and_r() {
    let (x, y) = two_part_fixture();
    let (x_pos, y_pos) = positive_rows(&x, &y);
    let input = ModelInput::new(x_pos, y_pos);
    let options = TweedieOptions::builder().robust_se(true).build();
    let (model, report) = fit_tweedie_input(&input, 2.0, options).expect("fit");

    let robust_se = column(report.se.as_ref().expect("robust SE"));
    let model_se = tweedie_model_based_se(&input.design_matrix, &model.beta, 2.0);
    assert_close(
        "tweedie(2) robust vs model-based",
        &robust_se,
        &model_se,
        0.1,
    );
    assert_close("tweedie(2) robust vs R", &robust_se, &R_HC0_GAMMA, 1e-6);
}

#[test]
fn tweedie_robust_se_matches_model_based_at_power_one_and_a_half() {
    // Gamma(shape = sqrt(mu), scale = sqrt(mu)) has mean mu and variance mu^1.5,
    // the Tweedie variance function at power 1.5 with unit dispersion.
    let mut stream = Stream(20_261_003);
    let mut rng = StdRng::seed_from_u64(20_261_003);
    let mut x = Mat::<f64>::zeros(N_ROWS, 3);
    let mut y = Mat::<f64>::zeros(N_ROWS, 1);
    for i in 0..N_ROWS {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = 2.0f64.mul_add(stream.uniform(), -1.0);
        x[(i, 2)] = if stream.uniform() < 0.3 { 1.0 } else { 0.0 };
        let mu = 0.3f64.mul_add(x[(i, 1)], 2.0 + 0.2 * x[(i, 2)]).exp();
        y[(i, 0)] = Gamma::new(mu.sqrt(), mu.sqrt())
            .expect("gamma parameters")
            .sample(&mut rng);
    }
    let input = ModelInput::new(x, y);
    let options = TweedieOptions::builder().robust_se(true).build();
    let (model, report) = fit_tweedie_input(&input, 1.5, options).expect("fit");
    let robust_se = column(report.se.as_ref().expect("robust SE"));
    let model_se = tweedie_model_based_se(&input.design_matrix, &model.beta, 1.5);
    assert_close(
        "tweedie(1.5) robust vs model-based",
        &robust_se,
        &model_se,
        0.1,
    );

    let clusters: Vec<u64> = (0..N_ROWS as u64).collect();
    let (_, clustered) =
        fit_tweedie_input(&input.with_cluster_ids(clusters), 1.5, options).expect("clustered fit");
    assert_close(
        "tweedie(1.5) singleton clusters",
        &column(clustered.se.as_ref().expect("SE")),
        &robust_se,
        1e-9,
    );
}
