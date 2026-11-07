//! The compact fits against the dense fits, and the cases where a compact fit refuses.
//!
//! Compact and dense fits take the same steps with the arithmetic in another order, so
//! they are compared within a tolerance: `AGREEMENT` for the gamma fits, relative to
//! `1 + |dense value|`, and `LOGIT_AGREEMENT` for the logit part. The dense path with
//! caller-held buffers is compared bit for bit with the dense path without them.

use faer::Mat;
use semicontinuous_models::models::FitStrategy;
use semicontinuous_models::{
    FitOptions, ModelInput, PreparedFitError, PreparedGammaDesign, PreparedPredictionDesign,
    PreparedScratch, PreparedTwoPartDesign, Regularization, TweedieOptions, TweedieScratch,
    TwoPartWarmStart, fit_tweedie_ref_warm_prevalidated,
    fit_tweedie_ref_warm_prevalidated_with_scratch, fit_two_part_ref_warm_prevalidated,
};

/// Largest accepted move between the compact and the dense fit of one weight column.
const AGREEMENT: f64 = 1e-8;

/// The same for the fitted logit linear predictors. The logit's last step is the size of
/// the convergence tolerance and its deviance change is the size of the rounding, so
/// either path can halve that step while the other takes it whole.
const LOGIT_AGREEMENT: f64 = 1e-5;

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

    fn exponential(&mut self) -> f64 {
        -self.uniform().max(1e-12).ln()
    }
}

/// Shaped like a matched cost cohort: an intercept, a calendar year, an age with two
/// spline-like terms, dummies from common to rare, and a heavy-tailed cost.
fn design(n: usize, zero_share: f64, seed: u64) -> (Mat<f64>, Mat<f64>) {
    let mut stream = Stream(seed);
    let p = 16;
    let mut x = Mat::<f64>::zeros(n, p);
    let mut y = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        let age = (stream.uniform() * stream.uniform() * 215.0).floor();
        x[(i, 1)] = age;
        x[(i, 2)] = (age / 60.0).powi(3);
        x[(i, 3)] = ((age - 12.0).max(0.0) / 60.0).powi(3);
        x[(i, 4)] = 2000.0 + (stream.uniform() * 19.0).floor();
        for (j, share) in (5..p).zip([
            0.6, 0.45, 0.3, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005,
        ]) {
            x[(i, j)] = f64::from(stream.uniform() < share);
        }
        let eta = 9.5 + 0.2 * x[(i, 5)] - 0.3 * x[(i, 6)] + 0.5 * x[(i, 12)] - 0.004 * age
            + 0.01 * (x[(i, 4)] - 2010.0);
        let noise = (1.2 * (stream.uniform() - 0.5) + 0.5 * (stream.exponential() - 1.0)).exp();
        y[(i, 0)] = if stream.uniform() < zero_share {
            0.0
        } else {
            eta.exp() * noise
        };
    }
    (x, y)
}

fn replicate_weights(n: usize, seed: u64) -> Mat<f64> {
    let mut stream = Stream(seed);
    Mat::from_fn(n, 1, |_, _| stream.exponential())
}

fn close(label: &str, compact: f64, dense: f64) {
    let relative = (compact - dense).abs() / (1.0 + dense.abs());
    assert!(
        relative <= AGREEMENT,
        "{label}: compact {compact} against dense {dense}, {relative:e} apart"
    );
}

fn close_columns(label: &str, compact: &Mat<f64>, dense: &Mat<f64>) {
    assert_eq!(compact.nrows(), dense.nrows(), "{label}");
    for j in 0..dense.nrows() {
        close(&format!("{label}[{j}]"), compact[(j, 0)], dense[(j, 0)]);
    }
}

/// The fitted linear predictors over the rows of `x`. The dense logit runs on the raw
/// design, whose intercept and calendar-year coefficients trade off against each other
/// by far more than either moves a prediction.
fn close_predictors(
    label: &str,
    x: &Mat<f64>,
    compact: &Mat<f64>,
    dense: &Mat<f64>,
    tolerance: f64,
) {
    let mut largest = 0.0_f64;
    for i in 0..x.nrows() {
        let difference: f64 = (0..x.ncols())
            .map(|j| x[(i, j)] * (compact[(j, 0)] - dense[(j, 0)]))
            .sum();
        largest = largest.max(difference.abs());
    }
    assert!(
        largest <= tolerance,
        "{label}: linear predictors {largest:e} apart"
    );
}

fn gamma_options() -> TweedieOptions {
    TweedieOptions {
        inner_parallel: false,
        ..TweedieOptions::default()
    }
}

fn two_part_options() -> FitOptions {
    FitOptions {
        compute_covariance: false,
        inner_parallel: false,
        ..FitOptions::default()
    }
}

fn gamma_warm(x: &Mat<f64>, y: &Mat<f64>) -> Mat<f64> {
    let input = ModelInput::new(x.clone(), y.clone());
    fit_tweedie_ref_warm_prevalidated(input.as_ref(), 2.0, gamma_options(), None)
        .expect("point fit")
        .0
        .beta
}

fn two_part_warm(x: &Mat<f64>, y: &Mat<f64>) -> TwoPartWarmStart {
    let input = ModelInput::new(x.clone(), y.clone());
    let (model, _) = fit_two_part_ref_warm_prevalidated(input.as_ref(), two_part_options(), None)
        .expect("point");
    TwoPartWarmStart {
        beta_logit: model.beta_logit,
        beta_gamma: model.beta_gamma,
    }
}

#[test]
fn a_gamma_fit_agrees_with_the_dense_fit() {
    let (x, y) = design(20_000, 0.0, 1);
    let warm = gamma_warm(&x, &y);
    let prepared = PreparedGammaDesign::new(x.as_ref(), y.col_as_slice(0)).expect("design");
    assert_eq!((prepared.nrows(), prepared.ncols()), (20_000, 16));
    let mut scratch = PreparedScratch::default();
    for seed in 0..12 {
        let weights = replicate_weights(20_000, 100 + seed);
        let input = ModelInput::new(x.clone(), y.clone()).with_sample_weights(weights.clone());
        let (dense, report) =
            fit_tweedie_ref_warm_prevalidated(input.as_ref(), 2.0, gamma_options(), Some(&warm))
                .expect("dense");
        let compact = prepared
            .fit(
                weights.col_as_slice(0),
                &warm,
                gamma_options(),
                &mut scratch,
            )
            .expect("compact");
        close_columns("beta", &compact.beta, &dense.beta);
        assert!(compact.iterations.abs_diff(report.meta.iterations) <= 1);
    }
}

#[test]
fn a_ridge_gamma_fit_agrees_with_the_dense_fit() {
    let (x, y) = design(8_000, 0.0, 2);
    let warm = gamma_warm(&x, &y);
    let options = TweedieOptions {
        l2_penalty: 1e-2,
        ..gamma_options()
    };
    let prepared = PreparedGammaDesign::new(x.as_ref(), y.col_as_slice(0)).expect("design");
    let weights = replicate_weights(8_000, 7);
    let input = ModelInput::new(x.clone(), y.clone()).with_sample_weights(weights.clone());
    let (dense, _) = fit_tweedie_ref_warm_prevalidated(input.as_ref(), 2.0, options, Some(&warm))
        .expect("dense");
    let compact = prepared
        .fit(
            weights.col_as_slice(0),
            &warm,
            options,
            &mut PreparedScratch::default(),
        )
        .expect("compact");
    close_columns("beta", &compact.beta, &dense.beta);
}

#[test]
fn a_two_part_fit_agrees_with_the_dense_fit() {
    for (zero_share, n) in [(0.35, 20_000), (0.02, 20_000)] {
        let (x, y) = design(n, zero_share, 3);
        let warm = two_part_warm(&x, &y);
        let prepared = PreparedTwoPartDesign::new(x.as_ref(), y.col_as_slice(0)).expect("design");
        assert_eq!(prepared.nrows(), n);
        assert!(prepared.positive_nrows() < n);
        let mut scratch = PreparedScratch::default();
        for seed in 0..8 {
            let weights = replicate_weights(n, 200 + seed);
            let input = ModelInput::new(x.clone(), y.clone()).with_sample_weights(weights.clone());
            let (dense, report) =
                fit_two_part_ref_warm_prevalidated(input.as_ref(), two_part_options(), Some(&warm))
                    .expect("dense");
            let compact = prepared
                .fit(
                    weights.col_as_slice(0),
                    &warm,
                    two_part_options(),
                    &mut scratch,
                )
                .expect("compact");
            close_predictors(
                "logit",
                &x,
                &compact.beta_logit,
                &dense.beta_logit,
                LOGIT_AGREEMENT,
            );
            close_predictors(
                "gamma",
                &x,
                &compact.beta_gamma,
                &dense.beta_gamma,
                AGREEMENT,
            );
            assert!(compact.iterations_logit.abs_diff(report.iterations_logit) <= 1);
            assert!(compact.iterations_gamma.abs_diff(report.iterations_gamma) <= 1);
        }
    }
}

#[test]
fn a_rare_outcome_class_gives_the_intercept_only_logit_of_the_dense_fit() {
    let (x, mut y) = design(6_000, 0.0, 4);
    // Fewer zeros than columns, so the dense logit takes the prevalence model.
    for i in 0..5 {
        y[(i * 100, 0)] = 0.0;
    }
    let warm = two_part_warm(&x, &y);
    let prepared = PreparedTwoPartDesign::new(x.as_ref(), y.col_as_slice(0)).expect("design");
    let weights = replicate_weights(6_000, 9);
    let input = ModelInput::new(x.clone(), y.clone()).with_sample_weights(weights.clone());
    let (dense, _) =
        fit_two_part_ref_warm_prevalidated(input.as_ref(), two_part_options(), Some(&warm))
            .expect("dense");
    let compact = prepared
        .fit(
            weights.col_as_slice(0),
            &warm,
            two_part_options(),
            &mut PreparedScratch::default(),
        )
        .expect("compact");
    assert_eq!(compact.iterations_logit, 0);
    close_columns("logit", &compact.beta_logit, &dense.beta_logit);
    close_predictors(
        "gamma",
        &x,
        &compact.beta_gamma,
        &dense.beta_gamma,
        AGREEMENT,
    );
}

#[test]
fn a_ridge_two_part_fit_agrees_with_the_dense_fit() {
    let (x, y) = design(8_000, 0.3, 5);
    let warm = two_part_warm(&x, &y);
    let options = FitOptions {
        regularization: Regularization::Ridge {
            lambda: 1e-2,
            exclude_intercept: true,
        },
        ..two_part_options()
    };
    let prepared = PreparedTwoPartDesign::new(x.as_ref(), y.col_as_slice(0)).expect("design");
    let weights = replicate_weights(8_000, 11);
    let input = ModelInput::new(x.clone(), y.clone()).with_sample_weights(weights.clone());
    let (dense, _) =
        fit_two_part_ref_warm_prevalidated(input.as_ref(), options, Some(&warm)).expect("dense");
    let compact = prepared
        .fit(
            weights.col_as_slice(0),
            &warm,
            options,
            &mut PreparedScratch::default(),
        )
        .expect("compact");
    close_predictors(
        "logit",
        &x,
        &compact.beta_logit,
        &dense.beta_logit,
        LOGIT_AGREEMENT,
    );
    close_predictors(
        "gamma",
        &x,
        &compact.beta_gamma,
        &dense.beta_gamma,
        AGREEMENT,
    );
}

#[test]
fn predictions_average_as_the_models_do() {
    let (x, y) = design(9_000, 0.25, 6);
    let gamma_y = Mat::from_fn(9_000, 1, |i, _| y[(i, 0)].max(1.0));
    let beta = gamma_warm(&x, &gamma_y);
    let input = ModelInput::new(x.clone(), gamma_y);
    let (gamma_model, _) =
        fit_tweedie_ref_warm_prevalidated(input.as_ref(), 2.0, gamma_options(), Some(&beta))
            .expect("gamma");
    let input = ModelInput::new(x.clone(), y);
    let (two_part_model, _) =
        fit_two_part_ref_warm_prevalidated(input.as_ref(), two_part_options(), None)
            .expect("two-part");

    let prediction = PreparedPredictionDesign::new(x.as_ref()).expect("design");
    assert_eq!(prediction.nrows(), 9_000);
    let mut stream = Stream(13);
    let weights: Vec<f64> = (0..9_000)
        .map(|i| {
            if i % 3 == 0 {
                0.0
            } else {
                stream.exponential()
            }
        })
        .collect();
    close(
        "gamma",
        prediction.mean_gamma(&gamma_model.beta, &weights),
        gamma_model.predict_mean(&x, Some(&weights)),
    );
    close(
        "two-part",
        prediction.mean_two_part(
            &two_part_model.beta_logit,
            &two_part_model.beta_gamma,
            &weights,
        ),
        two_part_model.predict_mean(&x, Some(&weights)),
    );
}

#[test]
fn a_prediction_design_may_hold_a_constant_column() {
    let (mut x, y) = design(2_000, 0.0, 7);
    let input = ModelInput::new(x.clone(), y.clone());
    let (model, _) =
        fit_tweedie_ref_warm_prevalidated(input.as_ref(), 2.0, gamma_options(), None).expect("fit");
    for i in 0..2_000 {
        x[(i, 5)] = 1.0;
    }
    assert!(PreparedGammaDesign::new(x.as_ref(), y.col_as_slice(0)).is_none());
    let weights = vec![1.0; 2_000];
    close(
        "mean",
        PreparedPredictionDesign::new(x.as_ref())
            .expect("design")
            .mean_gamma(&model.beta, &weights),
        model.predict_mean(&x, Some(&weights)),
    );
}

#[test]
fn a_design_the_compact_form_cannot_represent_is_refused() {
    let (x, y) = design(1_000, 0.2, 8);
    let y_gamma: Vec<f64> = y.col_as_slice(0).iter().map(|v| v.max(1.0)).collect();
    let y_two_part = y.col_as_slice(0);
    let gamma = |x: &Mat<f64>| PreparedGammaDesign::new(x.as_ref(), &y_gamma).is_some();
    let two_part = |x: &Mat<f64>| PreparedTwoPartDesign::new(x.as_ref(), y_two_part).is_some();
    let prediction = |x: &Mat<f64>| PreparedPredictionDesign::new(x.as_ref()).is_some();
    assert!(gamma(&x) && two_part(&x) && prediction(&x));

    // A constant column stays uncentred in the dense fit, and the algebra cannot.
    let mut constant = x.clone();
    for i in 0..1_000 {
        constant[(i, 3)] = 7.0;
    }
    assert!(!gamma(&constant) && !two_part(&constant) && prediction(&constant));

    // Column 0 is not the intercept.
    let mut no_intercept = x.clone();
    no_intercept[(10, 0)] = 2.0;
    assert!(!gamma(&no_intercept) && !two_part(&no_intercept) && !prediction(&no_intercept));

    // A column's number has to fit a byte.
    let wide = |p: usize| {
        Mat::from_fn(600, p, |i, j| {
            if j == 0 {
                1.0
            } else {
                f64::from(((i * 2_654_435_761 + j * 40_503) >> 7) % 5 == 0)
            }
        })
    };
    let y_wide = vec![1.0; 600];
    assert!(PreparedGammaDesign::new(wide(255).as_ref(), &y_wide).is_some());
    assert!(PreparedGammaDesign::new(wide(256).as_ref(), &y_wide).is_none());
    assert!(PreparedPredictionDesign::new(wide(255).as_ref()).is_some());
    assert!(PreparedPredictionDesign::new(wide(256).as_ref()).is_none());

    // The gamma deviance needs positive outcomes.
    assert!(PreparedGammaDesign::new(x.as_ref(), y_two_part).is_none());
    // A two-part design needs a positive row, and no column constant among those rows.
    let none_positive = vec![0.0; 1_000];
    assert!(PreparedTwoPartDesign::new(x.as_ref(), &none_positive).is_none());
    let mut constant_on_positive = x.clone();
    for i in 0..1_000 {
        constant_on_positive[(i, 3)] = if y_two_part[i] > 0.0 { 1.0 } else { 0.0 };
    }
    assert!(!two_part(&constant_on_positive));

    // Empty input and a length mismatch.
    let empty = Mat::<f64>::zeros(0, 3);
    assert!(PreparedPredictionDesign::new(empty.as_ref()).is_none());
    assert!(PreparedGammaDesign::new(x.as_ref(), &y_gamma[..10]).is_none());
    assert!(PreparedTwoPartDesign::new(x.as_ref(), &y_gamma[..10]).is_none());
}

#[test]
fn a_fit_refuses_what_it_cannot_answer_reliably() {
    let (x, y) = design(6_000, 0.0, 9);
    let warm = gamma_warm(&x, &y);
    let mut scratch = PreparedScratch::default();
    let prepared = PreparedGammaDesign::new(x.as_ref(), y.col_as_slice(0)).expect("design");
    let good = vec![1.0; 6_000];

    let mut bad = good.clone();
    bad[3] = 0.0;
    let error = |result: Result<(), PreparedFitError>| result.unwrap_err();
    assert_eq!(
        error(
            prepared
                .fit(&bad, &warm, gamma_options(), &mut scratch)
                .map(|_| ())
        ),
        PreparedFitError::InvalidWeights
    );
    assert_eq!(
        error(
            prepared
                .fit(&good[1..], &warm, gamma_options(), &mut scratch)
                .map(|_| ())
        ),
        PreparedFitError::InvalidWeights
    );
    assert_eq!(
        error(
            prepared
                .fit(&good, &Mat::zeros(3, 1), gamma_options(), &mut scratch)
                .map(|_| ())
        ),
        PreparedFitError::InvalidWarmStart
    );

    // No iterations allowed: the fit cannot have converged.
    let no_iterations = TweedieOptions {
        max_iter: 0,
        ..gamma_options()
    };
    assert_eq!(
        error(
            prepared
                .fit(&good, &warm, no_iterations, &mut scratch)
                .map(|_| ())
        ),
        PreparedFitError::NonConvergence
    );

    // Two identical columns: the pivot of the second is rounding only.
    let mut duplicated = x.clone();
    for i in 0..6_000 {
        duplicated[(i, 9)] = duplicated[(i, 8)];
    }
    let warm_zero = {
        let mut beta = Mat::zeros(16, 1);
        beta[(0, 0)] = 9.0;
        beta
    };
    let prepared_duplicated =
        PreparedGammaDesign::new(duplicated.as_ref(), y.col_as_slice(0)).expect("design");
    assert_eq!(
        error(
            prepared_duplicated
                .fit(&good, &warm_zero, gamma_options(), &mut scratch)
                .map(|_| ())
        ),
        PreparedFitError::IllConditioned
    );

    // A dummy that holds a 1 on all but one row, whose weight then almost vanishes.
    let mut nearly_all = x.clone();
    for i in 0..6_000 {
        nearly_all[(i, 9)] = f64::from(i != 17);
    }
    let prepared_nearly_all =
        PreparedGammaDesign::new(nearly_all.as_ref(), y.col_as_slice(0)).expect("design");
    let mut tiny = good.clone();
    tiny[17] = 1e-9;
    assert_eq!(
        error(
            prepared_nearly_all
                .fit(&tiny, &warm_zero, gamma_options(), &mut scratch)
                .map(|_| ())
        ),
        PreparedFitError::IllConditioned
    );

    // An outcome far beyond the log-link bound.
    let hot: Vec<f64> = y.col_as_slice(0).iter().map(|v| v * 1e20).collect();
    let prepared_hot = PreparedGammaDesign::new(x.as_ref(), &hot).expect("design");
    assert_eq!(
        error(
            prepared_hot
                .fit(&good, &warm_zero, gamma_options(), &mut scratch)
                .map(|_| ())
        ),
        PreparedFitError::Saturated
    );
}

#[test]
fn a_two_part_fit_refuses_what_it_cannot_answer_reliably() {
    let (x, y) = design(6_000, 0.3, 10);
    let warm = two_part_warm(&x, &y);
    let mut scratch = PreparedScratch::default();
    let prepared = PreparedTwoPartDesign::new(x.as_ref(), y.col_as_slice(0)).expect("design");
    let good = vec![1.0; 6_000];
    let code = |options: FitOptions,
                weights: &[f64],
                warm: &TwoPartWarmStart,
                scratch: &mut PreparedScratch| {
        prepared
            .fit(weights, warm, options, scratch)
            .map(|_| ())
            .unwrap_err()
    };

    assert_eq!(
        code(
            FitOptions {
                regularization: Regularization::ElasticNet {
                    lambda: 1e-3,
                    alpha: 0.5,
                    exclude_intercept: true
                },
                ..two_part_options()
            },
            &good,
            &warm,
            &mut scratch
        ),
        PreparedFitError::UnsupportedRegularization
    );
    assert_eq!(
        code(
            FitOptions {
                regularization: Regularization::Ridge {
                    lambda: 1e-3,
                    exclude_intercept: false
                },
                ..two_part_options()
            },
            &good,
            &warm,
            &mut scratch
        ),
        PreparedFitError::UnsupportedRegularization
    );
    let mut bad = good.clone();
    bad[0] = f64::NAN;
    assert_eq!(
        code(two_part_options(), &bad, &warm, &mut scratch),
        PreparedFitError::InvalidWeights
    );
    assert_eq!(
        code(
            two_part_options(),
            &good,
            &TwoPartWarmStart {
                beta_logit: Mat::zeros(2, 1),
                beta_gamma: Mat::zeros(16, 1),
            },
            &mut scratch
        ),
        PreparedFitError::InvalidWarmStart
    );
    assert_eq!(
        code(
            FitOptions {
                max_iter: 0,
                ..two_part_options()
            },
            &good,
            &warm,
            &mut scratch
        ),
        PreparedFitError::NonConvergence
    );
}

#[test]
fn a_compact_result_does_not_depend_on_what_the_scratch_held() {
    let (x, y) = design(6_000, 0.0, 11);
    let (x_two, y_two) = design(7_000, 0.3, 12);
    let warm = gamma_warm(&x, &y);
    let warm_two = two_part_warm(&x_two, &y_two);
    let gamma = PreparedGammaDesign::new(x.as_ref(), y.col_as_slice(0)).expect("gamma");
    let two_part = PreparedTwoPartDesign::new(x_two.as_ref(), y_two.col_as_slice(0)).expect("two");
    let weights = replicate_weights(6_000, 300);
    let weights_two = replicate_weights(7_000, 301);

    let fresh_gamma = gamma
        .fit(
            weights.col_as_slice(0),
            &warm,
            gamma_options(),
            &mut PreparedScratch::default(),
        )
        .expect("gamma");
    let fresh_two = two_part
        .fit(
            weights_two.col_as_slice(0),
            &warm_two,
            two_part_options(),
            &mut PreparedScratch::default(),
        )
        .expect("two-part");

    let mut shared = PreparedScratch::default();
    for _ in 0..2 {
        let used_two = two_part
            .fit(
                weights_two.col_as_slice(0),
                &warm_two,
                two_part_options(),
                &mut shared,
            )
            .expect("two-part");
        let used_gamma = gamma
            .fit(weights.col_as_slice(0), &warm, gamma_options(), &mut shared)
            .expect("gamma");
        for j in 0..16 {
            assert_eq!(
                fresh_gamma.beta[(j, 0)].to_bits(),
                used_gamma.beta[(j, 0)].to_bits()
            );
            assert_eq!(
                fresh_two.beta_logit[(j, 0)].to_bits(),
                used_two.beta_logit[(j, 0)].to_bits()
            );
            assert_eq!(
                fresh_two.beta_gamma[(j, 0)].to_bits(),
                used_two.beta_gamma[(j, 0)].to_bits()
            );
        }
    }
}

fn assert_same_bits(label: &str, a: &Mat<f64>, b: &Mat<f64>) {
    assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()), "{label}");
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            assert_eq!(
                a[(i, j)].to_bits(),
                b[(i, j)].to_bits(),
                "{label} ({i}, {j})"
            );
        }
    }
}

/// Every power on the path the scratch touches, against the fit that allocates, with one
/// scratch shared across fits of different shapes.
#[test]
fn a_tweedie_fit_with_a_scratch_is_bit_identical_to_one_without() {
    let (x, y) = design(7_000, 0.2, 13);
    let (x_small, y_small) = design(1_500, 0.2, 14);
    let positive = Mat::from_fn(7_000, 1, |i, _| y[(i, 0)].max(1.0));
    let weights = replicate_weights(7_000, 400);
    let weights_small = replicate_weights(1_500, 401);
    let relaxed = FitStrategy::Relaxed {
        fallback_lambda: 1e-3,
        max_retries: 3,
        warm_start: true,
        time_budget: None,
    };
    let mut scratch = TweedieScratch::default();
    for power in [1.0_f64, 1.2, 1.5, 1.8, 2.0] {
        let outcome = if (power - 2.0).abs() < 1e-12 {
            &positive
        } else {
            &y
        };
        for (label, options) in [
            ("default", gamma_options()),
            (
                "relaxed",
                TweedieOptions {
                    strategy: relaxed,
                    ..gamma_options()
                },
            ),
            (
                "ridge/robust",
                TweedieOptions {
                    l2_penalty: 1e-3,
                    robust_se: true,
                    ..gamma_options()
                },
            ),
        ] {
            let input =
                ModelInput::new(x.clone(), outcome.clone()).with_sample_weights(weights.clone());
            let (plain, plain_report) =
                fit_tweedie_ref_warm_prevalidated(input.as_ref(), power, options, None)
                    .expect("plain");
            let (held, held_report) = fit_tweedie_ref_warm_prevalidated_with_scratch(
                input.as_ref(),
                power,
                options,
                None,
                &mut scratch,
            )
            .expect("held");
            let name = format!("power {power} {label}");
            assert_same_bits(&name, &plain.beta, &held.beta);
            assert_eq!(
                plain_report.meta.iterations, held_report.meta.iterations,
                "{name}"
            );
            match (&plain_report.cov, &held_report.cov) {
                (Some(a), Some(b)) => assert_same_bits(&name, a, b),
                (None, None) => {}
                _ => panic!("{name}: one fit has a covariance"),
            }

            // Another shape reallocates; the next fit is unaffected by it.
            let outcome_small = Mat::from_fn(1_500, 1, |i, _| {
                if (power - 2.0).abs() < 1e-12 {
                    y_small[(i, 0)].max(1.0)
                } else {
                    y_small[(i, 0)]
                }
            });
            let small = ModelInput::new(x_small.clone(), outcome_small)
                .with_sample_weights(weights_small.clone());
            let (plain_small, _) =
                fit_tweedie_ref_warm_prevalidated(small.as_ref(), power, options, None)
                    .expect("plain");
            let (held_small, _) = fit_tweedie_ref_warm_prevalidated_with_scratch(
                small.as_ref(),
                power,
                options,
                None,
                &mut scratch,
            )
            .expect("held");
            assert_same_bits(&name, &plain_small.beta, &held_small.beta);
        }
    }
}

/// A design with no intercept is fitted uncentred, and leaves the scratch's centred
/// matrix untouched.
#[test]
fn a_scratch_serves_an_uncentred_fit_after_a_centred_one() {
    let (x, y) = design(3_000, 0.0, 15);
    let shifted = Mat::from_fn(3_000, x.ncols(), |i, j| {
        if j == 0 {
            1.0 + (i % 3) as f64 * 0.5
        } else {
            x[(i, j)]
        }
    });
    let mut scratch = TweedieScratch::default();
    let centred = ModelInput::new(x.clone(), y.clone());
    fit_tweedie_ref_warm_prevalidated_with_scratch(
        centred.as_ref(),
        2.0,
        gamma_options(),
        None,
        &mut scratch,
    )
    .expect("centred");
    let uncentred = ModelInput::new(shifted, y);
    let plain = fit_tweedie_ref_warm_prevalidated(uncentred.as_ref(), 2.0, gamma_options(), None);
    let held = fit_tweedie_ref_warm_prevalidated_with_scratch(
        uncentred.as_ref(),
        2.0,
        gamma_options(),
        None,
        &mut scratch,
    );
    match (plain, held) {
        (Ok((a, _)), Ok((b, _))) => assert_same_bits("uncentred", &a.beta, &b.beta),
        (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
        (a, b) => panic!("one fit failed: {:?} {:?}", a.err(), b.err()),
    }
}
