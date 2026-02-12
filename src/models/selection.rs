/////////////////////////////////////////////////////////////////////////////////////////////\
//
// Model selection utilities for semi-continuous outcome data.
//
// Created on: 24 Jan 2026     Author: Tobias Kragholm
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # Selection workflow
//!
//! Provides Park test diagnostics and goodness-of-fit metrics for choosing
//! between Tweedie GLM and log-normal models.

use faer::Mat;

use crate::models::matrix_ops::{select_rows, select_values};
use faer::prelude::Solve;
use rand::prelude::*;
use thiserror::Error;

use super::lognormal::{
    LogNormalError, LogNormalModel, LogNormalOptions, fit_lognormal_smearing,
    fit_lognormal_smearing_input, log_likelihood as lognormal_log_likelihood,
};
use super::tweedie::{
    TweedieError, TweedieModel, TweedieOptions, deviance as tweedie_deviance, fit_tweedie,
    fit_tweedie_input, quasi_log_likelihood as tweedie_quasi_log_likelihood,
};
use super::two_part::{FitOptions, TwoPartError, fit_two_part_input};
use crate::input::{InputError, ModelInput};

/// Goodness-of-fit metrics for model comparison.
#[derive(Debug, Clone)]
pub struct ModelFitMetrics {
    /// Root mean squared error.
    pub rmse: f64,
    /// Mean absolute error.
    pub mae: f64,
    /// Root mean squared log error (log1p scale).
    pub rmsle: f64,
    /// Coefficient of determination.
    pub r2: f64,
    /// Deviance or proxy deviance.
    pub deviance: f64,
}

/// Park test coefficients (variance function slope).
#[derive(Debug, Clone)]
pub struct ParkTestResult {
    pub intercept: f64,
    pub slope: f64,
}

/// Tweedie model candidate with metrics.
#[derive(Debug, Clone)]
pub struct TweedieCandidate {
    pub power: f64,
    pub model: TweedieModel,
    pub metrics: ModelFitMetrics,
    pub information_criteria: InformationCriteria,
}

/// Log-normal model candidate with metrics.
#[derive(Debug, Clone)]
pub struct LogNormalCandidate {
    pub model: LogNormalModel,
    pub metrics: ModelFitMetrics,
    pub information_criteria: InformationCriteria,
}

/// Combined model selection output.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    pub park_test: ParkTestResult,
    pub tweedie_candidates: Vec<TweedieCandidate>,
    pub lognormal_candidate: Option<LogNormalCandidate>,
    pub recommended_by_aic: Option<String>,
    pub recommended_by_bic: Option<String>,
}

/// Information criteria values for a fitted model.
#[derive(Debug, Clone, Copy)]
pub struct InformationCriteria {
    pub loglik: f64,
    pub aic: f64,
    pub bic: f64,
}

/// Tweedie cross-validation entry.
#[derive(Debug, Clone)]
pub struct TweedieCvCandidate {
    pub power: f64,
    pub metrics: ModelFitMetrics,
}

/// Cross-validation summary for candidate models.
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    pub folds_used: usize,
    pub two_part_metrics: ModelFitMetrics,
    pub tweedie_candidates: Vec<TweedieCvCandidate>,
    pub lognormal_metrics: Option<ModelFitMetrics>,
}

/// Options for K-fold cross-validation.
#[derive(Debug, Clone, Copy)]
pub struct CrossValidationOptions {
    pub k_folds: usize,
    pub seed: u64,
    pub two_part_options: FitOptions,
    pub tweedie_options: TweedieOptions,
    pub lognormal_options: LogNormalOptions,
}

impl Default for CrossValidationOptions {
    fn default() -> Self {
        Self {
            k_folds: 5,
            seed: 42,
            two_part_options: FitOptions::default(),
            tweedie_options: TweedieOptions::default(),
            lognormal_options: LogNormalOptions::default(),
        }
    }
}

/// Errors returned by cross-validation.
#[derive(Debug, Error)]
pub enum CrossValidationError {
    #[error("invalid model input: {0}")]
    Input(#[from] InputError),
    #[error("k-fold must be at least 2 and no more than the number of rows")]
    InvalidFolds,
    #[error("two-part fit failed: {0}")]
    TwoPart(#[from] TwoPartError),
    #[error("tweedie fit failed: {0}")]
    Tweedie(#[from] TweedieError),
    #[error("log-normal fit failed: {0}")]
    LogNormal(#[from] LogNormalError),
}

/// Compute the Park test slope from a gamma-log auxiliary regression.
#[must_use]
pub fn park_test(x: &Mat<f64>, y: &Mat<f64>) -> Option<ParkTestResult> {
    if y.ncols() != 1 || x.nrows() != y.nrows() {
        return None;
    }
    let positive_indices: Vec<usize> = (0..y.nrows()).filter(|&idx| y[(idx, 0)] > 0.0).collect();
    if positive_indices.is_empty() {
        return None;
    }

    let x_pos = select_rows(x, &positive_indices);
    let y_pos = select_values(y, &positive_indices);

    let (gamma, _report) =
        fit_tweedie(&x_pos, &y_pos, None, 2.0, TweedieOptions::default()).ok()?;
    let mu = gamma.predict(&x_pos).mean;
    let residuals = Mat::from_fn(y_pos.nrows(), 1, |i, _| y_pos[(i, 0)] - mu[(i, 0)]);
    let log_resid = Mat::from_fn(residuals.nrows(), 1, |i, _| {
        (residuals[(i, 0)].powi(2)).ln()
    });
    let log_mu = Mat::from_fn(mu.nrows(), 1, |i, _| mu[(i, 0)].ln());

    let design = Mat::from_fn(
        log_mu.nrows(),
        2,
        |i, j| if j == 0 { 1.0 } else { log_mu[(i, 0)] },
    );
    let xtx = design.transpose() * &design;
    let xty = design.transpose() * &log_resid;
    let beta = xtx.full_piv_lu().solve(xty);

    Some(ParkTestResult {
        intercept: beta[(0, 0)],
        slope: beta[(1, 0)],
    })
}

/// Fit candidate Tweedie powers and a log-normal model, returning metrics.
#[must_use]
pub(crate) fn select_models(x: &Mat<f64>, y: &Mat<f64>, powers: &[f64]) -> SelectionResult {
    let park = park_test(x, y).unwrap_or(ParkTestResult {
        intercept: 0.0,
        slope: 2.0,
    });

    let mut tweedie_candidates = Vec::new();
    for &power in powers {
        if let Ok((model, _report)) = fit_tweedie(x, y, None, power, TweedieOptions::default()) {
            let pred = model.predict(x);
            let metrics = compute_model_fit_metrics(y, &pred.mean, Some(power));
            let ll = tweedie_quasi_log_likelihood(y, &pred.mean, power);
            let ic = compute_information_criteria(ll, x.ncols(), y.nrows());
            tweedie_candidates.push(TweedieCandidate {
                power,
                model,
                metrics,
                information_criteria: ic,
            });
        }
    }

    let lognormal_candidate = fit_lognormal_smearing(x, y, None, LogNormalOptions::default())
        .ok()
        .map(|(model, _report)| {
            let pred = model.predict(x);
            let metrics = compute_model_fit_metrics(y, &pred.mean, None);
            let ll = lognormal_log_likelihood(x, y, &model);
            let ic = compute_information_criteria(ll, x.ncols(), y.nrows());
            LogNormalCandidate {
                model,
                metrics,
                information_criteria: ic,
            }
        });

    let (aic_recommendation, bic_recommendation) =
        recommend_by_ic(&tweedie_candidates, lognormal_candidate.as_ref());

    SelectionResult {
        park_test: park,
        tweedie_candidates,
        lognormal_candidate,
        recommended_by_aic: aic_recommendation,
        recommended_by_bic: bic_recommendation,
    }
}

/// Run model selection for a `ModelInput` container.
#[must_use]
///
/// # Examples
///
/// ```
/// use faer::Mat;
/// use semicontinuous_models::{ModelInput, select_models_input};
///
/// fn idx_to_f64(idx: usize) -> f64 {
///     f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
/// }
///
/// let n = 50;
/// let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 15.0 });
/// let y = Mat::from_fn(n, 1, |i, _| if i % 5 == 0 { 0.0 } else { 0.2f64.mul_add(idx_to_f64(i), 1.5) });
/// let input = ModelInput::new(x, y);
///
/// let result = select_models_input(&input, &[1.2, 1.5]);
/// assert!(!result.tweedie_candidates.is_empty());
/// ```
pub fn select_models_input(input: &ModelInput, powers: &[f64]) -> SelectionResult {
    select_models(&input.design_matrix, &input.outcome, powers)
}

/// Compute RMSE, MAE, and a deviance-like score.
#[must_use]
pub fn compute_model_fit_metrics(
    y: &Mat<f64>,
    mu: &Mat<f64>,
    tweedie_power: Option<f64>,
) -> ModelFitMetrics {
    if y.nrows() == 0 || y.ncols() != 1 || mu.ncols() != 1 || y.nrows() != mu.nrows() {
        return ModelFitMetrics {
            rmse: f64::NAN,
            mae: f64::NAN,
            rmsle: f64::NAN,
            r2: f64::NAN,
            deviance: f64::NAN,
        };
    }
    let mut mse = 0.0;
    let mut mae = 0.0;
    let mut log_error_sq_sum = 0.0;
    let mut sst = 0.0;
    let mut mean_y = 0.0;
    let denom = f64::from(u32::try_from(y.nrows()).unwrap_or(u32::MAX));
    for i in 0..y.nrows() {
        mean_y += y[(i, 0)];
    }
    mean_y /= denom.max(1.0);

    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        let mui = mu[(i, 0)];
        let diff = yi - mui;
        mse += diff * diff;
        mae += diff.abs();
        let log_diff = yi.max(0.0).ln_1p() - mui.max(0.0).ln_1p();
        log_error_sq_sum += log_diff * log_diff;
        let dev = yi - mean_y;
        sst += dev * dev;
    }
    let rmse = (mse / denom).sqrt();
    let mae = mae / denom;
    let rms_log_error = (log_error_sq_sum / denom).sqrt();
    let r2 = if sst > 0.0 { 1.0 - (mse / sst) } else { 0.0 };
    let deviance = tweedie_power.map_or(rmse * rmse, |power| tweedie_deviance(y, mu, power));

    ModelFitMetrics {
        rmse,
        mae,
        rmsle: rms_log_error,
        r2,
        deviance,
    }
}

/// Compute AIC/BIC for a given log-likelihood, parameter count, and sample size.
#[must_use]
pub fn compute_information_criteria(loglik: f64, k: usize, n: usize) -> InformationCriteria {
    let k_f = f64::from(u32::try_from(k).unwrap_or(u32::MAX));
    let n_f = f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    let aic = (-2.0f64).mul_add(loglik, 2.0 * k_f);
    let bic = if n > 1 {
        (-2.0f64).mul_add(loglik, n_f.ln() * k_f)
    } else {
        f64::NAN
    };
    InformationCriteria { loglik, aic, bic }
}

/// Recommend a model based on CV RMSE (lower is better).
#[must_use]
pub fn recommend_from_cv(cv: &CrossValidationResult) -> Option<String> {
    let mut best_name: Option<String> = None;
    let mut best_rmse = f64::INFINITY;

    if cv.two_part_metrics.rmse < best_rmse {
        best_rmse = cv.two_part_metrics.rmse;
        best_name = Some("two_part".to_string());
    }

    for candidate in &cv.tweedie_candidates {
        if candidate.metrics.rmse < best_rmse {
            best_rmse = candidate.metrics.rmse;
            best_name = Some(format!("tweedie p={:.1}", candidate.power));
        }
    }

    if let Some(lognormal) = &cv.lognormal_metrics
        && lognormal.rmse < best_rmse
    {
        best_name = Some("lognormal".to_string());
    }

    best_name
}

/// Run K-fold cross-validation for two-part, Tweedie candidates, and log-normal models.
///
/// Notes:
/// - Weights and cluster labels are used for the two-part model (via `ModelInput`).
/// - Tweedie/log-normal fits ignore weights, but use clusters for robust SEs when enabled.
///
/// # Errors
///
/// Returns `CrossValidationError` if any fold fit fails or `k_folds` is invalid.
pub fn cross_validate_models_input(
    input: &ModelInput,
    powers: &[f64],
    options: CrossValidationOptions,
) -> Result<CrossValidationResult, CrossValidationError> {
    input.validate()?;
    let n = input.outcome.nrows();
    if options.k_folds < 2 || options.k_folds > n {
        return Err(CrossValidationError::InvalidFolds);
    }

    let mut indices = (0..n).collect::<Vec<_>>();
    let mut rng = rand::rngs::StdRng::seed_from_u64(options.seed);
    indices.shuffle(&mut rng);

    let fold_size = n.div_ceil(options.k_folds);
    let mut folds_used = 0usize;

    let mut two_part_accum = zero_metrics();
    let mut lognormal_accum = zero_metrics();
    let mut tweedie_accum = powers.iter().map(|_| zero_metrics()).collect::<Vec<_>>();

    for fold in 0..options.k_folds {
        let start = fold * fold_size;
        let end = (start + fold_size).min(n);
        if start >= end {
            continue;
        }
        let test_idx = &indices[start..end];
        let train_idx = complement_indices(n, test_idx);

        let train_input = subset_input(input, &train_idx);
        let test_input = subset_input(input, test_idx);

        let two_part = fit_two_part_input(&train_input, options.two_part_options)?;
        let pred_two_part = two_part.0.predict(&test_input.design_matrix);
        let metrics_two_part =
            compute_model_fit_metrics(&test_input.outcome, &pred_two_part.expected_outcome, None);
        add_metrics(&mut two_part_accum, &metrics_two_part);

        for (pos, power) in powers.iter().copied().enumerate() {
            let (tweedie, _report) =
                fit_tweedie_input(&train_input, power, options.tweedie_options)?;
            let pred = tweedie.predict(&test_input.design_matrix);
            let metrics = compute_model_fit_metrics(&test_input.outcome, &pred.mean, Some(power));
            add_metrics(&mut tweedie_accum[pos], &metrics);
        }

        let (lognormal, _report) =
            fit_lognormal_smearing_input(&train_input, options.lognormal_options)?;
        let pred_lognormal = lognormal.predict(&test_input.design_matrix);
        let metrics_lognormal =
            compute_model_fit_metrics(&test_input.outcome, &pred_lognormal.mean, None);
        add_metrics(&mut lognormal_accum, &metrics_lognormal);

        folds_used += 1;
    }

    let folds_f = f64::from(u32::try_from(folds_used).unwrap_or(u32::MAX));
    if folds_used > 0 {
        scale_metrics(&mut two_part_accum, folds_f);
        scale_metrics(&mut lognormal_accum, folds_f);
        for metrics in &mut tweedie_accum {
            scale_metrics(metrics, folds_f);
        }
    }

    let tweedie_candidates = powers
        .iter()
        .copied()
        .zip(tweedie_accum)
        .map(|(power, metrics)| TweedieCvCandidate { power, metrics })
        .collect();

    let result = CrossValidationResult {
        folds_used,
        two_part_metrics: two_part_accum,
        tweedie_candidates,
        lognormal_metrics: Some(lognormal_accum),
    };
    Ok(result)
}

const fn zero_metrics() -> ModelFitMetrics {
    ModelFitMetrics {
        rmse: 0.0,
        mae: 0.0,
        rmsle: 0.0,
        r2: 0.0,
        deviance: 0.0,
    }
}

fn add_metrics(accum: &mut ModelFitMetrics, next: &ModelFitMetrics) {
    accum.rmse += next.rmse;
    accum.mae += next.mae;
    accum.rmsle += next.rmsle;
    accum.r2 += next.r2;
    accum.deviance += next.deviance;
}

fn scale_metrics(metrics: &mut ModelFitMetrics, folds: f64) {
    metrics.rmse /= folds;
    metrics.mae /= folds;
    metrics.rmsle /= folds;
    metrics.r2 /= folds;
    metrics.deviance /= folds;
}

fn subset_input(input: &ModelInput, indices: &[usize]) -> ModelInput {
    let x = select_rows(&input.design_matrix, indices);
    let y = select_rows(&input.outcome, indices);
    let weights = input
        .sample_weights
        .as_ref()
        .map(|w| select_rows(w, indices));
    let clusters = input
        .cluster_ids
        .as_ref()
        .map(|labels| indices.iter().map(|&idx| labels[idx]).collect());
    let mut subset = ModelInput::new(x, y);
    if let Some(sample_weights) = weights {
        subset = subset.with_sample_weights(sample_weights);
    }
    if let Some(cluster_ids) = clusters {
        subset = subset.with_cluster_ids(cluster_ids);
    }
    subset
}

fn complement_indices(n: usize, test_idx: &[usize]) -> Vec<usize> {
    let mut mask = vec![false; n];
    for &idx in test_idx {
        mask[idx] = true;
    }
    (0..n).filter(|&idx| !mask[idx]).collect()
}

fn recommend_by_ic(
    tweedie_candidates: &[TweedieCandidate],
    lognormal: Option<&LogNormalCandidate>,
) -> (Option<String>, Option<String>) {
    let mut best_aic: Option<(f64, String)> = None;
    let mut best_bic_candidate: Option<(f64, String)> = None;

    for candidate in tweedie_candidates {
        let name = format!("tweedie p={:.1}", candidate.power);
        best_aic = pick_best(best_aic, candidate.information_criteria.aic, &name);
        best_bic_candidate = pick_best(
            best_bic_candidate,
            candidate.information_criteria.bic,
            &name,
        );
    }

    if let Some(lognormal) = lognormal {
        best_aic = pick_best(best_aic, lognormal.information_criteria.aic, "lognormal");
        best_bic_candidate = pick_best(
            best_bic_candidate,
            lognormal.information_criteria.bic,
            "lognormal",
        );
    }

    (
        best_aic.map(|(_, name)| name),
        best_bic_candidate.map(|(_, name)| name),
    )
}

fn pick_best(current: Option<(f64, String)>, value: f64, name: &str) -> Option<(f64, String)> {
    match current {
        Some((best, _)) if value >= best => current,
        _ => Some((value, name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn idx_to_f64(idx: usize) -> f64 {
        f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
    }

    #[test]
    fn selection_runs() {
        let n = 60;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 });
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 6 == 0 {
                0.0
            } else {
                0.2f64.mul_add(idx_to_f64(i), 1.0)
            }
        });
        let result = select_models(&x, &y, &[1.2, 1.5, 1.8]);
        assert!(!result.tweedie_candidates.is_empty());
        assert!(result.park_test.slope.is_finite());
    }

    #[test]
    fn cross_validate_runs() {
        let n = 50;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 });
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 7 == 0 {
                0.0
            } else {
                0.2f64.mul_add(idx_to_f64(i), 1.0)
            }
        });
        let input = ModelInput::new(x, y);
        let result =
            cross_validate_models_input(&input, &[1.2, 1.5], CrossValidationOptions::default())
                .expect("cv");
        assert_eq!(result.tweedie_candidates.len(), 2);
        assert!(result.folds_used > 0);
    }

    #[test]
    fn compute_metrics_returns_nan_on_dimension_mismatch() {
        let y = Mat::from_fn(2, 1, |i, _| if i == 0 { 0.0 } else { 1.0 });
        let mu = Mat::from_fn(3, 1, |_, _| 1.0);
        let metrics = compute_model_fit_metrics(&y, &mu, None);
        assert!(metrics.rmse.is_nan());
        assert!(metrics.deviance.is_nan());
    }

    #[test]
    fn bic_is_nan_when_sample_size_is_too_small() {
        let ic = compute_information_criteria(-10.0, 3, 1);
        assert!(ic.bic.is_nan());
    }

    #[test]
    fn cross_validate_rejects_invalid_fold_counts() {
        let n = 8;
        let design_matrix =
            Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 });
        let outcome = Mat::from_fn(n, 1, |i, _| if i % 3 == 0 { 0.0 } else { 1.0 });
        let input = ModelInput::new(design_matrix, outcome);

        let low = cross_validate_models_input(
            &input,
            &[1.5],
            CrossValidationOptions {
                k_folds: 1,
                ..CrossValidationOptions::default()
            },
        )
        .expect_err("k_folds < 2 should fail");
        assert!(matches!(low, CrossValidationError::InvalidFolds));

        let high = cross_validate_models_input(
            &input,
            &[1.5],
            CrossValidationOptions {
                k_folds: n + 1,
                ..CrossValidationOptions::default()
            },
        )
        .expect_err("k_folds > n should fail");
        assert!(matches!(high, CrossValidationError::InvalidFolds));
    }

    #[test]
    fn cross_validate_allows_empty_power_grid() {
        let n = 30;
        let design_matrix =
            Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 });
        let outcome = Mat::from_fn(n, 1, |i, _| if i % 5 == 0 { 0.0 } else { 1.0 });
        let input = ModelInput::new(design_matrix, outcome);
        let result = cross_validate_models_input(&input, &[], CrossValidationOptions::default())
            .expect("cv should run without Tweedie powers");
        assert!(result.tweedie_candidates.is_empty());
    }

    #[test]
    fn cross_validate_is_deterministic_for_fixed_seed() {
        let n = 40;
        let design_matrix =
            Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 8.0 });
        let outcome = Mat::from_fn(n, 1, |i, _| if i % 6 == 0 { 0.0 } else { 1.0 });
        let input = ModelInput::new(design_matrix, outcome);
        let options = CrossValidationOptions {
            k_folds: 4,
            seed: 12345,
            ..CrossValidationOptions::default()
        };
        let first = cross_validate_models_input(&input, &[1.2, 1.5], options).expect("first run");
        let second = cross_validate_models_input(&input, &[1.2, 1.5], options).expect("second run");
        assert_eq!(first.folds_used, second.folds_used);
        assert_eq!(
            first.tweedie_candidates.len(),
            second.tweedie_candidates.len()
        );
        assert_relative_eq!(first.two_part_metrics.rmse, second.two_part_metrics.rmse);
        assert_relative_eq!(first.two_part_metrics.mae, second.two_part_metrics.mae);
        assert_relative_eq!(first.two_part_metrics.rmsle, second.two_part_metrics.rmsle);
        assert_relative_eq!(first.two_part_metrics.r2, second.two_part_metrics.r2);
        assert_relative_eq!(
            first.two_part_metrics.deviance,
            second.two_part_metrics.deviance
        );
    }

    #[test]
    fn cross_validate_propagates_two_part_model_errors() {
        let n = 20;
        let design_matrix =
            Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 5.0 });
        let outcome = Mat::from_fn(n, 1, |_i, _| 0.0);
        let input = ModelInput::new(design_matrix, outcome);
        let err = cross_validate_models_input(&input, &[], CrossValidationOptions::default())
            .expect_err("all-zero outcomes should fail two-part fitting");
        assert!(matches!(err, CrossValidationError::TwoPart(_)));
    }
}
