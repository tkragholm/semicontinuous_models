/////////////////////////////////////////////////////////////////////////////////////////////\
//
// Tweedie GLM (quasi-likelihood) for semi-continuous outcome data.
//
// Created on: 24 Jan 2026     Author: Tobias Kragholm
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # Tweedie GLM
//!
//! Implements a Tweedie GLM with log link using quasi-likelihood IRLS.
//! The Tweedie power (p) should be in [1, 2] to allow exact zeros with
//! continuous positive outcomes.

use std::collections::HashMap;

use faer::Mat;

use crate::input::{InputError, ModelInput};
use crate::models::matrix_ops::map_mat;
use crate::utils::{max_abs_diff, solve_linear_system};

/// Tuning parameters for Tweedie fitting.
#[derive(Debug, Clone, Copy)]
pub struct TweedieOptions {
    /// Maximum IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Lower bound on IRLS weights.
    pub min_weight: f64,
    /// L2 (ridge) penalty strength.
    pub l2_penalty: f64,
    /// If true, do not penalize the first column (intercept).
    pub l2_penalty_exclude_intercept: bool,
    /// If true, compute robust (sandwich) covariance.
    pub robust_se: bool,
}

impl Default for TweedieOptions {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tolerance: 1e-6,
            min_weight: 1e-6,
            l2_penalty: 0.0,
            l2_penalty_exclude_intercept: true,
            robust_se: false,
        }
    }
}

/// Errors returned by Tweedie fitting.
#[derive(Debug, thiserror::Error)]
pub enum TweedieError {
    #[error("design matrix rows ({rows}) must match outcome length ({len})")]
    DimensionMismatch { rows: usize, len: usize },
    #[error("design matrix must have at least one column")]
    EmptyDesign,
    #[error("tweedie power must be between 1 and 2")]
    InvalidPower,
    #[error("outcome must be a single column matrix")]
    InvalidOutcomeShape,
    #[error("cluster labels length ({labels}) must match outcome rows ({rows})")]
    InvalidClusterLength { labels: usize, rows: usize },
    #[error("inputs contain non-finite values")]
    NonFiniteInput,
    #[error("outcome contains negative values")]
    NegativeOutcome,
    #[error("model failed to converge")]
    NonConvergence,
    #[error("linear solve failed")]
    SolveFailed,
}

impl From<InputError> for TweedieError {
    fn from(value: InputError) -> Self {
        match value {
            InputError::EmptyDesign => Self::EmptyDesign,
            InputError::InvalidOutcomeShape => Self::InvalidOutcomeShape,
            InputError::DimensionMismatch { rows, len } => Self::DimensionMismatch { rows, len },
            InputError::InvalidClusterLength { labels, rows } => {
                Self::InvalidClusterLength { labels, rows }
            }
            InputError::NonFiniteDesign
            | InputError::NonFiniteOutcome
            | InputError::InvalidWeightShape
            | InputError::NonFiniteWeights
            | InputError::NonPositiveWeights => Self::NonFiniteInput,
            InputError::NegativeOutcome => Self::NegativeOutcome,
        }
    }
}

/// Tweedie model coefficients.
#[derive(Debug, Clone)]
pub struct TweedieModel {
    /// Coefficient vector.
    pub beta: Mat<f64>,
    /// Tweedie power parameter (p).
    pub power: f64,
}

/// Tweedie diagnostics and inference outputs.
#[derive(Debug, Clone)]
pub struct TweedieReport {
    /// Standard errors for coefficients.
    pub se: Option<Mat<f64>>,
    /// Covariance matrix for coefficients.
    pub cov: Option<Mat<f64>>,
    /// True if robust (sandwich) covariance was used.
    pub robust: bool,
    /// True if cluster-robust covariance was used.
    pub clustered: bool,
    /// Number of clusters used for cluster-robust covariance.
    pub cluster_count: Option<usize>,
}

/// Tweedie predictions (mean on original scale).
#[derive(Debug, Clone)]
pub struct TweediePrediction {
    pub mean: Mat<f64>,
}

impl TweedieModel {
    /// Predict the mean outcome on the original scale.
    #[must_use]
    pub fn predict(&self, x: &Mat<f64>) -> TweediePrediction {
        let eta = x * &self.beta;
        let mean = map_mat(&eta, f64::exp);
        TweediePrediction { mean }
    }
}

/// Fit a Tweedie GLM with log link using quasi-likelihood IRLS.
///
/// # Errors
///
/// Returns `TweedieError` if inputs are malformed or the solver fails.
pub(crate) fn fit_tweedie(
    x: &Mat<f64>,
    y: &Mat<f64>,
    clusters: Option<&[u64]>,
    power: f64,
    options: TweedieOptions,
) -> Result<(TweedieModel, TweedieReport), TweedieError> {
    if !(1.0..=2.0).contains(&power) {
        return Err(TweedieError::InvalidPower);
    }
    if x.ncols() == 0 {
        return Err(TweedieError::EmptyDesign);
    }
    if y.ncols() != 1 {
        return Err(TweedieError::InvalidOutcomeShape);
    }
    if x.nrows() != y.nrows() {
        return Err(TweedieError::DimensionMismatch {
            rows: x.nrows(),
            len: y.nrows(),
        });
    }
    if let Some(cluster_ids) = clusters
        && cluster_ids.len() != y.nrows()
    {
        return Err(TweedieError::InvalidClusterLength {
            labels: cluster_ids.len(),
            rows: y.nrows(),
        });
    }
    if !crate::utils::matrix_is_finite(x) || !crate::utils::matrix_is_finite(y) {
        return Err(TweedieError::NonFiniteInput);
    }
    if (0..y.nrows()).any(|i| y[(i, 0)] < 0.0) {
        return Err(TweedieError::NegativeOutcome);
    }

    let mut beta = Mat::<f64>::zeros(x.ncols(), 1);
    let lambda = options.l2_penalty.max(0.0);

    for _ in 0..options.max_iter {
        let eta = x * &beta;
        let mu = map_mat(&eta, f64::exp);

        let weights = Mat::from_fn(mu.nrows(), 1, |i, _| {
            let w = mu[(i, 0)].powf(2.0 - power);
            w.max(options.min_weight)
        });

        let z = Mat::from_fn(mu.nrows(), 1, |i, _| {
            eta[(i, 0)] + (y[(i, 0)] - mu[(i, 0)]) / mu[(i, 0)]
        });

        let mut xtwx = weighted_xtx(x, &weights);
        if lambda > 0.0 {
            xtwx += ridge_penalty(x.ncols(), lambda, options.l2_penalty_exclude_intercept);
        }
        let xtw_rhs = weighted_xtz(x, &weights, &z);
        let beta_next =
            solve_linear_system(&xtwx, &xtw_rhs).map_err(|_| TweedieError::SolveFailed)?;

        if max_abs_diff(&beta_next, &beta) < options.tolerance {
            let beta_final = beta_next;
            let eta = x * &beta_final;
            let mu = map_mat(&eta, f64::exp);
            let weights = Mat::from_fn(mu.nrows(), 1, |i, _| {
                let w = mu[(i, 0)].powf(2.0 - power);
                w.max(options.min_weight)
            });
            let mut xtwx = weighted_xtx(x, &weights);
            if lambda > 0.0 {
                xtwx += ridge_penalty(x.ncols(), lambda, options.l2_penalty_exclude_intercept);
            }
            let (cov, se, clustered, cluster_count) = if options.robust_se {
                let xtwx_inv = invert_matrix(&xtwx)?;
                let (cov, clustered, cluster_count) =
                    robust_covariance(x, y, &mu, &weights, clusters, &xtwx_inv);
                let se = cov.as_ref().map(diagonal_sqrt);
                (cov, se, clustered, cluster_count)
            } else {
                (None, None, false, None)
            };
            let model = TweedieModel {
                beta: beta_final,
                power,
            };
            let report = TweedieReport {
                se,
                cov,
                robust: options.robust_se,
                clustered,
                cluster_count,
            };
            return Ok((model, report));
        }
        beta = beta_next;
    }

    Err(TweedieError::NonConvergence)
}

/// Fit a Tweedie GLM from a `ModelInput` container.
///
/// # Errors
///
/// Returns `TweedieError` if inputs are malformed or the solver fails.
///
/// # Examples
///
/// ```
/// use faer::Mat;
/// use semicontinuous_models::{ModelInput, TweedieOptions, fit_tweedie_input};
///
/// fn idx_to_f64(idx: usize) -> f64 {
///     f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
/// }
///
/// let n = 40;
/// let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 20.0 });
/// let y = Mat::from_fn(n, 1, |i, _| if i % 5 == 0 { 0.0 } else { 0.2f64.mul_add(idx_to_f64(i), 2.0) });
/// let input = ModelInput::new(x, y);
///
/// let (model, _report) = fit_tweedie_input(&input, 1.5, TweedieOptions::default()).expect("fit");
/// assert_eq!(model.beta.nrows(), 2);
/// ```
pub fn fit_tweedie_input(
    input: &ModelInput,
    power: f64,
    options: TweedieOptions,
) -> Result<(TweedieModel, TweedieReport), TweedieError> {
    input.validate_core().map_err(TweedieError::from)?;
    fit_tweedie(
        &input.design_matrix,
        &input.outcome,
        input.cluster_ids.as_deref(),
        power,
        options,
    )
}

/// Compute Tweedie deviance for a fitted mean and power.
#[must_use]
pub fn deviance(y: &Mat<f64>, mu: &Mat<f64>, power: f64) -> f64 {
    if y.ncols() != 1 || mu.ncols() != 1 || y.nrows() != mu.nrows() {
        return f64::NAN;
    }
    if !power.is_finite() || !(1.0..=2.0).contains(&power) {
        return f64::NAN;
    }

    // Closed-form endpoint for p = 1 (Poisson deviance limit).
    if (power - 1.0).abs() < 1e-12 {
        let mut result = 0.0;
        for i in 0..y.nrows() {
            let yi = y[(i, 0)].max(0.0);
            let mui = mu[(i, 0)].max(1e-12);
            if yi == 0.0 {
                result += 2.0 * mui;
            } else {
                result += 2.0 * (yi * (yi / mui).ln() - (yi - mui));
            }
        }
        return result;
    }

    // Closed-form endpoint for p = 2 (Gamma deviance limit).
    if (power - 2.0).abs() < 1e-12 {
        let mut result = 0.0;
        for i in 0..y.nrows() {
            let yi = y[(i, 0)];
            let mui = mu[(i, 0)].max(1e-12);
            if yi <= 0.0 {
                return f64::INFINITY;
            }
            result += 2.0 * (((yi - mui) / mui) - (yi / mui).ln());
        }
        return result;
    }

    let mut deviance = 0.0;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        let mui = mu[(i, 0)].max(1e-12);
        if yi == 0.0 {
            deviance += 2.0 * mui.powf(2.0 - power) / (2.0 - power);
        } else {
            let term1 = yi.powf(2.0 - power) / ((1.0 - power) * (2.0 - power));
            let term2 = yi * mui.powf(1.0 - power) / (1.0 - power);
            let term3 = mui.powf(2.0 - power) / (2.0 - power);
            deviance += 2.0 * (term1 - term2 + term3);
        }
    }
    deviance
}

/// Compute a quasi log-likelihood from the Tweedie deviance.
///
/// This uses the common approximation `-0.5 * deviance` for model comparison
/// when the full Tweedie likelihood is not available.
#[must_use]
pub fn quasi_log_likelihood(y: &Mat<f64>, mu: &Mat<f64>, power: f64) -> f64 {
    let dev = deviance(y, mu, power);
    if dev.is_finite() { -0.5 * dev } else { dev }
}

fn invert_matrix(a: &Mat<f64>) -> Result<Mat<f64>, TweedieError> {
    let identity = Mat::from_fn(a.nrows(), a.ncols(), |i, j| if i == j { 1.0 } else { 0.0 });
    solve_linear_system(a, &identity).map_err(|_| TweedieError::SolveFailed)
}

fn diagonal_sqrt(cov: &Mat<f64>) -> Mat<f64> {
    Mat::from_fn(cov.nrows(), 1, |i, _| cov[(i, i)].max(0.0).sqrt())
}

fn robust_covariance(
    x: &Mat<f64>,
    y: &Mat<f64>,
    mu: &Mat<f64>,
    weights: &Mat<f64>,
    clusters: Option<&[u64]>,
    xtwx_inv: &Mat<f64>,
) -> (Option<Mat<f64>>, bool, Option<usize>) {
    let p = x.ncols();
    let mut meat = Mat::<f64>::zeros(p, p);
    if let Some(clusters) = clusters {
        let mut cluster_sums: HashMap<u64, Vec<f64>> = HashMap::new();
        for i in 0..x.nrows() {
            let resid = (y[(i, 0)] - mu[(i, 0)]) * weights[(i, 0)];
            let entry = cluster_sums
                .entry(clusters[i])
                .or_insert_with(|| vec![0.0; p]);
            for j in 0..p {
                entry[j] += resid * x[(i, j)];
            }
        }
        for sum in cluster_sums.values() {
            for j in 0..p {
                for k in 0..p {
                    meat[(j, k)] += sum[j] * sum[k];
                }
            }
        }
        let cov = xtwx_inv * &meat * xtwx_inv;
        return (Some(cov), true, Some(cluster_sums.len()));
    }

    for i in 0..x.nrows() {
        let resid = (y[(i, 0)] - mu[(i, 0)]) * weights[(i, 0)];
        for j in 0..p {
            let xj = x[(i, j)];
            for k in 0..p {
                meat[(j, k)] += resid * resid * xj * x[(i, k)];
            }
        }
    }
    let cov = xtwx_inv * &meat * xtwx_inv;
    (Some(cov), false, None)
}

fn ridge_penalty(ncols: usize, lambda: f64, exclude_intercept: bool) -> Mat<f64> {
    Mat::from_fn(ncols, ncols, |i, j| {
        if i == j {
            if exclude_intercept && i == 0 {
                0.0
            } else {
                lambda
            }
        } else {
            0.0
        }
    })
}

fn weighted_xtx(x: &Mat<f64>, weights: &Mat<f64>) -> Mat<f64> {
    let p = x.ncols();
    let mut xtx = Mat::<f64>::zeros(p, p);
    for i in 0..x.nrows() {
        let w = weights[(i, 0)];
        for col_i in 0..p {
            let wxi = w * x[(i, col_i)];
            for col_j in 0..p {
                xtx[(col_i, col_j)] += wxi * x[(i, col_j)];
            }
        }
    }
    xtx
}

fn weighted_xtz(x: &Mat<f64>, weights: &Mat<f64>, z: &Mat<f64>) -> Mat<f64> {
    let p = x.ncols();
    let mut xtz = Mat::<f64>::zeros(p, 1);
    for i in 0..x.nrows() {
        let wz = weights[(i, 0)] * z[(i, 0)];
        for col in 0..p {
            xtz[(col, 0)] += x[(i, col)] * wz;
        }
    }
    xtz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx_to_f64(idx: usize) -> f64 {
        f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
    }

    #[test]
    fn fit_tweedie_runs() {
        let n = 100;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 20.0 });
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 5 == 0 {
                0.0
            } else {
                0.2f64.mul_add(idx_to_f64(i), 2.0)
            }
        });
        let (model, _report) =
            fit_tweedie(&x, &y, None, 1.5, TweedieOptions::default()).expect("fit");
        let pred = model.predict(&x);
        assert_eq!(pred.mean.nrows(), n);
        let dev = deviance(&y, &pred.mean, 1.5);
        assert!(dev.is_finite());
    }

    #[test]
    fn tweedie_reports_robust_se_with_clusters() {
        let n = 40;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 20.0 });
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 5 == 0 {
                0.0
            } else {
                0.2f64.mul_add(idx_to_f64(i), 2.0)
            }
        });
        let clusters: Vec<u64> = (0..n).map(|i| (i / 4) as u64).collect();
        let options = TweedieOptions {
            robust_se: true,
            ..TweedieOptions::default()
        };
        let (model, report) = fit_tweedie(&x, &y, Some(&clusters), 1.5, options).expect("fit");
        let pred = model.predict(&x);
        assert_eq!(pred.mean.nrows(), n);
        assert!(report.robust);
        assert!(report.clustered);
        assert!(report.se.is_some());
    }

    #[test]
    fn deviance_returns_nan_on_shape_mismatch() {
        let y = Mat::from_fn(3, 1, |_i, _| 1.0);
        let mu = Mat::from_fn(2, 1, |_i, _| 1.0);
        let dev = deviance(&y, &mu, 1.5);
        assert!(dev.is_nan());
    }

    #[test]
    fn deviance_supports_boundary_power_one() {
        let y = Mat::from_fn(4, 1, |i, _| if i == 0 { 0.0 } else { 1.0 + idx_to_f64(i) });
        let mu = Mat::from_fn(4, 1, |_i, _| 1.5);
        let dev = deviance(&y, &mu, 1.0);
        assert!(dev.is_finite());
        let qll = quasi_log_likelihood(&y, &mu, 1.0);
        assert!(qll.is_finite());
    }

    #[test]
    fn deviance_supports_boundary_power_two_with_positive_outcomes() {
        let y = Mat::from_fn(4, 1, |i, _| 1.0 + idx_to_f64(i) / 2.0);
        let mu = Mat::from_fn(4, 1, |_i, _| 1.5);
        let dev = deviance(&y, &mu, 2.0);
        assert!(dev.is_finite());
        let qll = quasi_log_likelihood(&y, &mu, 2.0);
        assert!(qll.is_finite());
    }

    #[test]
    fn deviance_power_two_with_zeros_is_infinite() {
        let y = Mat::from_fn(3, 1, |i, _| if i == 0 { 0.0 } else { 1.0 });
        let mu = Mat::from_fn(3, 1, |_i, _| 1.0);
        let dev = deviance(&y, &mu, 2.0);
        assert!(dev.is_infinite());
    }
}
