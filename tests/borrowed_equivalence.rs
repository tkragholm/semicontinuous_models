//! The fits that borrow their input and the mean-only predictions give the same bits as
//! the owned fits and the full predictions.
//!
//! These tests compare two paths of the same binary, so unlike `fit_golden` they hold on
//! every target.

use faer::Mat;
use semicontinuous_models::{
    FitOptions, InputError, LogNormalOptions, Model, ModelInput, ModelInputRef, TweedieError,
    TweedieOptions, TwoPartError, TwoPartWarmStart, fit_lognormal_smearing_input,
    fit_tweedie_input_warm, fit_tweedie_ref_warm, fit_tweedie_ref_warm_prevalidated,
    fit_two_part_input_warm, fit_two_part_ref_warm, fit_two_part_ref_warm_prevalidated,
};

const N_COLS: usize = 8;

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

/// Intercept, five dummies and two continuous columns, with about 40% zeros.
fn sample(n: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
    let mut stream = Stream(seed);
    let mut x = Mat::<f64>::zeros(n, N_COLS);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        for j in 1..6 {
            x[(i, j)] = f64::from(stream.uniform() < 0.1 + 0.07 * j as f64);
        }
        for j in 6..N_COLS {
            x[(i, j)] = 2.0f64.mul_add(stream.uniform(), -1.0);
        }
    }
    let mut y = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        let eta = 0.4f64.mul_add(x[(i, 1)], 0.2 * x[(i, 2)] - 0.1 * x[(i, 6)]);
        let is_zero = stream.uniform() < 0.4 - 0.1 * x[(i, 3)];
        let noise = -stream.uniform().max(1e-12).ln() - stream.uniform().max(1e-12).ln();
        if !is_zero {
            y[(i, 0)] = (6.0f64 + eta).exp() / 2.0 * noise + 1.0;
        }
    }
    (x, y)
}

/// Weights of a fractional-weight replicate: positive and different for each replicate.
fn replicate_weights(n: usize, replicate: u64) -> Mat<f64> {
    let mut stream = Stream(7_000 + replicate);
    Mat::from_fn(n, 1, |_, _| 0.2 + 2.0 * stream.uniform())
}

fn same(a: &Mat<f64>, b: &Mat<f64>) {
    assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            assert_eq!(a[(i, j)].to_bits(), b[(i, j)].to_bits(), "entry ({i}, {j})");
        }
    }
}

fn same_opt(a: Option<&Mat<f64>>, b: Option<&Mat<f64>>) {
    match (a, b) {
        (Some(a), Some(b)) => same(a, b),
        (None, None) => {}
        _ => panic!("one side has the matrix and the other does not"),
    }
}

fn same_two_part(
    a: &(
        semicontinuous_models::TwoPartModel,
        semicontinuous_models::TwoPartReport,
    ),
    b: &(
        semicontinuous_models::TwoPartModel,
        semicontinuous_models::TwoPartReport,
    ),
) {
    same(&a.0.beta_logit, &b.0.beta_logit);
    same(&a.0.beta_gamma, &b.0.beta_gamma);
    same_opt(a.1.se_logit.as_ref(), b.1.se_logit.as_ref());
    same_opt(a.1.se_gamma.as_ref(), b.1.se_gamma.as_ref());
    same_opt(a.1.cov_logit.as_ref(), b.1.cov_logit.as_ref());
    same_opt(a.1.cov_gamma.as_ref(), b.1.cov_gamma.as_ref());
    assert_eq!(a.1.iterations_logit, b.1.iterations_logit);
    assert_eq!(a.1.iterations_gamma, b.1.iterations_gamma);
    assert_eq!(a.1.cluster_count, b.1.cluster_count);
}

fn same_tweedie(
    a: &(
        semicontinuous_models::TweedieModel,
        semicontinuous_models::TweedieReport,
    ),
    b: &(
        semicontinuous_models::TweedieModel,
        semicontinuous_models::TweedieReport,
    ),
) {
    same(&a.0.beta, &b.0.beta);
    same_opt(a.1.se.as_ref(), b.1.se.as_ref());
    same_opt(a.1.cov.as_ref(), b.1.cov.as_ref());
    assert_eq!(a.1.meta.iterations, b.1.meta.iterations);
    assert_eq!(a.1.cluster_count, b.1.cluster_count);
}

fn clusters(n: usize) -> Vec<u64> {
    (0..n as u64).map(|i| i / 3 * 7_919 + 13).collect()
}

#[test]
fn two_part_borrowed_fit_matches_owned_fit() {
    let n = 6_000;
    let (x, y) = sample(n, 11);
    let cluster_ids = clusters(n);
    let options = [
        FitOptions::default(),
        FitOptions::builder().robust_se(true).build(),
        FitOptions::builder().compute_covariance(false).build(),
    ];
    let full = fit_two_part_input_warm(
        &ModelInput::new(x.clone(), y.clone()).with_sample_weights(replicate_weights(n, 0)),
        FitOptions::default(),
        None,
    )
    .expect("full-sample fit");
    let warm = TwoPartWarmStart {
        beta_logit: full.0.beta_logit.clone(),
        beta_gamma: full.0.beta_gamma.clone(),
    };

    for options in options {
        for replicate in 1..=3_u64 {
            let weights = replicate_weights(n, replicate);
            for with_clusters in [false, true] {
                for warm in [None, Some(&warm)] {
                    let mut owned =
                        ModelInput::new(x.clone(), y.clone()).with_sample_weights(weights.clone());
                    let mut borrowed = ModelInputRef::new(x.as_ref(), y.as_ref())
                        .with_sample_weights(weights.as_ref());
                    if with_clusters {
                        owned = owned.with_cluster_ids(cluster_ids.clone());
                        borrowed = borrowed.with_cluster_ids(&cluster_ids);
                    }
                    let a = fit_two_part_input_warm(&owned, options, warm).expect("owned");
                    let b = fit_two_part_ref_warm(borrowed, options, warm).expect("borrowed");
                    let c = fit_two_part_ref_warm_prevalidated(borrowed, options, warm)
                        .expect("prevalidated");
                    same_two_part(&a, &b);
                    same_two_part(&a, &c);
                }
            }
        }
    }
}

#[test]
fn two_part_unweighted_borrowed_fit_matches_owned_fit() {
    let n = 5_000;
    let (x, y) = sample(n, 12);
    let owned = ModelInput::new(x.clone(), y.clone());
    let borrowed = ModelInputRef::new(x.as_ref(), y.as_ref());
    let options = FitOptions::default();
    let a = fit_two_part_input_warm(&owned, options, None).expect("owned");
    let b = fit_two_part_ref_warm(borrowed, options, None).expect("borrowed");
    same_two_part(&a, &b);
}

#[test]
fn tweedie_borrowed_fit_matches_owned_fit() {
    let n = 6_000;
    let (x, y) = sample(n, 13);
    let cluster_ids = clusters(n);
    let full = fit_tweedie_input_warm(
        &ModelInput::new(x.clone(), y.clone()).with_sample_weights(replicate_weights(n, 0)),
        2.0,
        TweedieOptions::default(),
        None,
    )
    .expect("full-sample fit");
    for power in [1.5, 2.0] {
        for options in [
            TweedieOptions::default(),
            TweedieOptions::builder().robust_se(true).build(),
        ] {
            for replicate in 1..=3_u64 {
                let weights = replicate_weights(n, replicate);
                for with_clusters in [false, true] {
                    for warm in [None, Some(&full.0.beta)] {
                        let mut owned = ModelInput::new(x.clone(), y.clone())
                            .with_sample_weights(weights.clone());
                        let mut borrowed = ModelInputRef::new(x.as_ref(), y.as_ref())
                            .with_sample_weights(weights.as_ref());
                        if with_clusters {
                            owned = owned.with_cluster_ids(cluster_ids.clone());
                            borrowed = borrowed.with_cluster_ids(&cluster_ids);
                        }
                        let a =
                            fit_tweedie_input_warm(&owned, power, options, warm).expect("owned");
                        let b =
                            fit_tweedie_ref_warm(borrowed, power, options, warm).expect("borrowed");
                        let c = fit_tweedie_ref_warm_prevalidated(borrowed, power, options, warm)
                            .expect("prevalidated");
                        same_tweedie(&a, &b);
                        same_tweedie(&a, &c);
                    }
                }
            }
        }
    }
}

/// A comparator subset gathered once and fitted with the weights of each replicate gives
/// the fit of the subset input rebuilt for each replicate.
#[test]
fn row_subset_gathered_once_matches_rebuilt_subset() {
    let n = 5_000;
    let (x, y) = sample(n, 14);
    let rows: Vec<usize> = (0..n).filter(|&i| x[(i, 6)] < 0.5).collect();
    let x_subset = Mat::from_fn(rows.len(), N_COLS, |i, j| x[(rows[i], j)]);
    let y_subset = Mat::from_fn(rows.len(), 1, |i, _| y[(rows[i], 0)]);
    let design = ModelInputRef::new(x_subset.as_ref(), y_subset.as_ref());
    for replicate in 1..=3_u64 {
        let all = replicate_weights(n, replicate);
        let weights = Mat::from_fn(rows.len(), 1, |i, _| all[(rows[i], 0)]);
        let rebuilt = ModelInput::new(x_subset.clone(), y_subset.clone())
            .with_sample_weights(weights.clone());
        let a = fit_two_part_input_warm(&rebuilt, FitOptions::default(), None).expect("rebuilt");
        let b = fit_two_part_ref_warm_prevalidated(
            design.with_sample_weights(weights.as_ref()),
            FitOptions::default(),
            None,
        )
        .expect("borrowed");
        same_two_part(&a, &b);
    }
}

#[test]
fn prevalidated_fits_still_check_the_weights_and_clusters() {
    let n = 200;
    let (x, y) = sample(n, 15);
    let mut weights = replicate_weights(n, 1);
    weights[(3, 0)] = 0.0;
    let input = ModelInputRef::new(x.as_ref(), y.as_ref()).with_sample_weights(weights.as_ref());
    assert!(matches!(
        fit_two_part_ref_warm_prevalidated(input, FitOptions::default(), None),
        Err(TwoPartError::NonPositiveWeights)
    ));
    assert!(matches!(
        fit_tweedie_ref_warm_prevalidated(input, 2.0, TweedieOptions::default(), None),
        Err(TweedieError::NonPositiveWeights)
    ));
    let short = vec![1_u64; n - 1];
    let input = ModelInputRef::new(x.as_ref(), y.as_ref()).with_cluster_ids(&short);
    assert!(fit_two_part_ref_warm_prevalidated(input, FitOptions::default(), None).is_err());
}

#[test]
fn borrowed_validation_matches_owned_validation() {
    let (x, y) = sample(50, 16);
    let mut bad_y = y.clone();
    bad_y[(2, 0)] = -1.0;
    let mut nan_x = x.clone();
    nan_x[(4, 3)] = f64::NAN;
    let weights = Mat::from_fn(50, 1, |i, _| if i == 7 { 0.0 } else { 1.0 });
    let cases = [
        ModelInput::new(x.clone(), y.clone()),
        ModelInput::new(x.clone(), bad_y),
        ModelInput::new(nan_x, y.clone()),
        ModelInput::new(x.clone(), y.clone()).with_sample_weights(weights),
        ModelInput::new(x.clone(), y.clone()).with_cluster_ids(vec![1; 49]),
        ModelInput::new(x.clone(), y).with_column_labels(vec!["a".into(); N_COLS]),
    ];
    let expected: [Result<(), InputError>; 6] = [
        Ok(()),
        Err(InputError::NegativeOutcome),
        Err(InputError::NonFiniteDesign),
        Err(InputError::NonPositiveWeights),
        Err(InputError::InvalidClusterLength {
            labels: 49,
            rows: 50,
        }),
        Err(InputError::DuplicateLabels("a".into())),
    ];
    for (case, expected) in cases.iter().zip(expected) {
        assert_eq!(case.validate(), expected);
        assert_eq!(case.as_ref().validate(), expected);
    }
}

fn weights_with_zeros(n: usize) -> Vec<f64> {
    let mut stream = Stream(99);
    (0..n)
        .map(|_| {
            let u = stream.uniform();
            if u < 0.3 { 0.0 } else { 3.0 * u }
        })
        .collect()
}

/// The mean as the downstream takes it: a plain sum over the rows, or the sum of the
/// weighted rows with the nonzero weights, in row order.
fn reference_mean(values: &Mat<f64>, weights: Option<&[f64]>) -> f64 {
    match weights {
        None => {
            let sum: f64 = (0..values.nrows()).map(|i| values[(i, 0)]).sum();
            sum / values.nrows() as f64
        }
        Some(weights) => {
            let (sum, total) = (0..values.nrows())
                .filter(|&i| weights[i] != 0.0)
                .fold((0.0, 0.0), |(sum, total), i| {
                    (sum + weights[i] * values[(i, 0)], total + weights[i])
                });
            sum / total
        }
    }
}

#[test]
fn predict_mean_matches_the_mean_of_the_full_prediction() {
    let (x_fit, y_fit) = sample(3_000, 21);
    let input = ModelInput::new(x_fit, y_fit);
    let (two_part, _) = fit_two_part_input_warm(&input, FitOptions::default(), None).expect("fit");
    let (tweedie, _) =
        fit_tweedie_input_warm(&input, 1.5, TweedieOptions::default(), None).expect("fit");
    let (lognormal, _) =
        fit_lognormal_smearing_input(&input, LogNormalOptions::default()).expect("fit");

    // Sizes around and across the block of 4096 rows the linear predictor is formed in.
    for n in [1, 2, 7, 100, 4_095, 4_096, 4_097, 8_192, 8_193, 25_000] {
        let (x, _) = sample(n, 1_000 + n as u64);
        let weights = weights_with_zeros(n);
        for weights in [None, Some(weights.as_slice())] {
            assert_eq!(
                two_part.predict_mean(&x, weights).to_bits(),
                reference_mean(&two_part.predict(&x).expected_outcome, weights).to_bits(),
                "two-part, n = {n}"
            );
            assert_eq!(
                tweedie.predict_mean(&x, weights).to_bits(),
                reference_mean(&tweedie.predict(&x).mean, weights).to_bits(),
                "tweedie, n = {n}"
            );
            assert_eq!(
                lognormal.predict_mean(&x, weights).to_bits(),
                reference_mean(&lognormal.predict(&x).mean, weights).to_bits(),
                "log-normal, n = {n}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "one weight per row")]
fn predict_mean_rejects_weights_of_the_wrong_length() {
    let (x, y) = sample(300, 22);
    let (model, _) = fit_tweedie_input_warm(
        &ModelInput::new(x.clone(), y),
        2.0,
        TweedieOptions::default(),
        None,
    )
    .expect("fit");
    let _ = model.predict_mean(&x, Some(&[1.0; 5]));
}
