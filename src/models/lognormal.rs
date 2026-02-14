/////////////////////////////////////////////////////////////////////////////////////////////\
//
// Log-normal regression with smearing retransformation.
//
// Created on: 24 Jan 2026     Author: Tobias Kragholm
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # Log-normal with smearing
//!
//! Fits a log-normal regression on positive outcomes and applies a smearing
//! factor for unbiased retransformation to the original scale.

use std::collections::HashMap;

use faer::Mat;

use crate::models::matrix_ops::{map_mat, select_rows, select_values};
use thiserror::Error;

use crate::input::{InputError, ModelInput};
use crate::utils::{max_abs_diff, mean_column, solve_linear_system};

/// Tuning parameters for log-normal fitting.
#[derive(Debug, Clone, Copy)]
pub struct LogNormalOptions {
    /// Maximum number of iterations (usually 1 for OLS).
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// L2 (ridge) penalty strength.
    pub l2_penalty: f64,
    /// If true, do not penalize the first column (intercept).
    pub l2_penalty_exclude_intercept: bool,
    /// If true, compute robust (sandwich) covariance.
    pub robust_se: bool,
}

impl Default for LogNormalOptions {
    fn default() -> Self {
        Self {
            max_iter: 1,
            tolerance: 1e-12,
            l2_penalty: 0.0,
            l2_penalty_exclude_intercept: true,
            robust_se: false,
        }
    }
}

/// Errors returned by log-normal fitting.
#[derive(Debug, Error)]
pub enum LogNormalError {
    #[error("design matrix rows ({rows}) must match outcome length ({len})")]
    DimensionMismatch { rows: usize, len: usize },
    #[error("design matrix must have at least one column")]
    EmptyDesign,
    #[error("outcome must be a single column matrix")]
    InvalidOutcomeShape,
    #[error("cluster labels length ({labels}) must match outcome rows ({rows})")]
    InvalidClusterLength { labels: usize, rows: usize },
    #[error("inputs contain non-finite values")]
    NonFiniteInput,
    #[error("outcome contains negative values")]
    NegativeOutcome,
    #[error("no positive outcomes")]
    NoPositiveOutcomes,
    #[error("linear solve failed")]
    SolveFailed,
}

impl From<InputError> for LogNormalError {
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

/// Log-normal model coefficients and smearing factor.
#[derive(Debug, Clone)]
pub struct LogNormalModel {
    /// Coefficient vector for log-outcomes.
    pub beta: Mat<f64>,
    /// Smearing factor for retransformation.
    pub smearing_factor: f64,
}

/// Model diagnostics and inference outputs.
#[derive(Debug, Clone)]
pub struct LogNormalReport {
    /// Standard errors for the log-outcome coefficients.
    pub se: Option<Mat<f64>>,
    /// Covariance matrix for the coefficients.
    pub cov: Option<Mat<f64>>,
    /// True if robust (sandwich) covariance was used.
    pub robust: bool,
    /// True if cluster-robust covariance was used.
    pub clustered: bool,
    /// Number of clusters used for cluster-robust covariance.
    pub cluster_count: Option<usize>,
}

/// Log-normal predictions on the original scale.
#[derive(Debug, Clone)]
pub struct LogNormalPrediction {
    pub mean: Mat<f64>,
}

type RobustCovarianceResult = (Option<Mat<f64>>, bool, Option<usize>);

impl LogNormalModel {
    /// Predict mean outcome on the original scale.
    #[must_use]
    pub fn predict(&self, x: &Mat<f64>) -> LogNormalPrediction {
        let eta = x * &self.beta;
        let mean = Mat::from_fn(eta.nrows(), 1, |i, _| {
            self.smearing_factor * eta[(i, 0)].exp()
        });
        LogNormalPrediction { mean }
    }
}

/// Compute the log-likelihood under a log-normal model (positive outcomes only).
///
/// Uses the fitted linear predictor for `log(y)` and estimates the residual
/// variance from the positive outcomes.
#[must_use]
pub fn log_likelihood(x: &Mat<f64>, y: &Mat<f64>, model: &LogNormalModel) -> f64 {
    if y.ncols() != 1 || x.nrows() != y.nrows() {
        return f64::NAN;
    }
    let eta = x * &model.beta;
    let mut sumsq = 0.0;
    let mut n_pos = 0.0;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        if yi > 0.0 {
            let resid = yi.ln() - eta[(i, 0)];
            sumsq += resid * resid;
            n_pos += 1.0;
        }
    }
    if n_pos == 0.0 {
        return f64::NAN;
    }
    let sigma2 = (sumsq / n_pos).max(1e-12);
    let mut loglik = 0.0;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        if yi > 0.0 {
            let resid = yi.ln() - eta[(i, 0)];
            loglik += (-0.5f64).mul_add(
                std::f64::consts::TAU.ln() + sigma2.ln() + resid * resid / sigma2,
                -yi.ln(),
            );
        }
    }
    loglik
}

/// Fit a log-normal regression with smearing retransformation.
///
/// # Errors
///
/// Returns `LogNormalError` if inputs are malformed or no positive outcomes exist.
pub(crate) fn fit_lognormal_smearing(
    x: &Mat<f64>,
    y: &Mat<f64>,
    clusters: Option<&[u64]>,
    options: LogNormalOptions,
) -> Result<(LogNormalModel, LogNormalReport), LogNormalError> {
    if x.ncols() == 0 {
        return Err(LogNormalError::EmptyDesign);
    }
    if y.ncols() != 1 {
        return Err(LogNormalError::InvalidOutcomeShape);
    }
    if x.nrows() != y.nrows() {
        return Err(LogNormalError::DimensionMismatch {
            rows: x.nrows(),
            len: y.nrows(),
        });
    }
    if let Some(cluster_ids) = clusters
        && cluster_ids.len() != y.nrows()
    {
        return Err(LogNormalError::InvalidClusterLength {
            labels: cluster_ids.len(),
            rows: y.nrows(),
        });
    }
    if !crate::utils::matrix_is_finite(x) || !crate::utils::matrix_is_finite(y) {
        return Err(LogNormalError::NonFiniteInput);
    }
    if (0..y.nrows()).any(|i| y[(i, 0)] < 0.0) {
        return Err(LogNormalError::NegativeOutcome);
    }

    let positive_indices: Vec<usize> = (0..y.nrows()).filter(|&idx| y[(idx, 0)] > 0.0).collect();

    if positive_indices.is_empty() {
        return Err(LogNormalError::NoPositiveOutcomes);
    }

    let x_pos = select_rows(x, &positive_indices);
    let y_pos = select_values(y, &positive_indices);
    let y_log = Mat::from_fn(y_pos.nrows(), 1, |i, _| y_pos[(i, 0)].ln());
    let cluster_ids_pos = clusters.map(|labels| {
        positive_indices
            .iter()
            .map(|&idx| labels[idx])
            .collect::<Vec<_>>()
    });

    let mut beta = Mat::<f64>::zeros(x_pos.ncols(), 1);
    let lambda = options.l2_penalty.max(0.0);

    for _ in 0..options.max_iter {
        let xtx = x_pos.transpose() * &x_pos
            + ridge_penalty(x_pos.ncols(), lambda, options.l2_penalty_exclude_intercept);
        let xty = x_pos.transpose() * &y_log;
        let beta_next = solve_linear_system(&xtx, &xty).map_err(|_| LogNormalError::SolveFailed)?;

        if max_abs_diff(&beta_next, &beta) < options.tolerance {
            beta = beta_next;
            break;
        }
        beta = beta_next;
    }

    let eta = &x_pos * &beta;
    let residuals = Mat::from_fn(y_log.nrows(), 1, |i, _| y_log[(i, 0)] - eta[(i, 0)]);
    let smearing = mean_column(&map_mat(&residuals, f64::exp));

    let (cov, se, clustered, cluster_count) = if options.robust_se {
        let xtx = x_pos.transpose() * &x_pos
            + ridge_penalty(x_pos.ncols(), lambda, options.l2_penalty_exclude_intercept);
        let (cov, clustered, cluster_count) =
            robust_covariance(&x_pos, &residuals, cluster_ids_pos.as_deref(), &xtx)?;
        let se = cov.as_ref().map(diagonal_sqrt);
        (cov, se, clustered, cluster_count)
    } else {
        (None, None, false, None)
    };

    let model = LogNormalModel {
        beta,
        smearing_factor: smearing,
    };
    let report = LogNormalReport {
        se,
        cov,
        robust: options.robust_se,
        clustered,
        cluster_count,
    };

    Ok((model, report))
}

/// Fit a log-normal model with smearing from a `ModelInput` container.
///
/// # Errors
///
/// Returns `LogNormalError` if inputs are malformed or no positive outcomes exist.
///
/// # Examples
///
/// ```
/// use faer::Mat;
/// use semicontinuous_models::{ModelInput, LogNormalOptions, fit_lognormal_smearing_input};
///
/// fn idx_to_f64(idx: usize) -> f64 {
///     f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
/// }
///
/// let n = 30;
/// let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 });
/// let y = Mat::from_fn(n, 1, |i, _| if i % 4 == 0 { 0.0 } else { 0.3f64.mul_add(idx_to_f64(i), 2.0) });
/// let input = ModelInput::new(x, y);
///
/// let (model, _report) =
///     fit_lognormal_smearing_input(&input, LogNormalOptions::default()).expect("fit");
/// assert!(model.smearing_factor > 0.0);
/// ```
pub fn fit_lognormal_smearing_input(
    input: &ModelInput,
    options: LogNormalOptions,
) -> Result<(LogNormalModel, LogNormalReport), LogNormalError> {
    input.validate_core().map_err(LogNormalError::from)?;
    fit_lognormal_smearing(
        &input.design_matrix,
        &input.outcome,
        input.cluster_ids.as_deref(),
        options,
    )
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

fn diagonal_sqrt(cov: &Mat<f64>) -> Mat<f64> {
    Mat::from_fn(cov.nrows(), 1, |i, _| cov[(i, i)].max(0.0).sqrt())
}

fn robust_covariance(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    clusters: Option<&[u64]>,
    xtx: &Mat<f64>,
) -> Result<RobustCovarianceResult, LogNormalError> {
    let p = x.ncols();
    let mut meat = Mat::<f64>::zeros(p, p);
    if let Some(clusters) = clusters {
        let mut cluster_sums: HashMap<u64, Vec<f64>> = HashMap::new();
        for i in 0..x.nrows() {
            let resid = residuals[(i, 0)];
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
        let cov = sandwich_covariance(xtx, &meat)?;
        return Ok((Some(cov), true, Some(cluster_sums.len())));
    }

    for i in 0..x.nrows() {
        let resid = residuals[(i, 0)];
        for j in 0..p {
            let xj = x[(i, j)];
            for k in 0..p {
                meat[(j, k)] += resid * resid * xj * x[(i, k)];
            }
        }
    }
    let cov = sandwich_covariance(xtx, &meat)?;
    Ok((Some(cov), false, None))
}

fn sandwich_covariance(xtx: &Mat<f64>, meat: &Mat<f64>) -> Result<Mat<f64>, LogNormalError> {
    let left = solve_linear_system(xtx, meat).map_err(|_| LogNormalError::SolveFailed)?;
    let xtx_t = transpose_owned(xtx);
    let left_t = transpose_owned(&left);
    let cov_t = solve_linear_system(&xtx_t, &left_t).map_err(|_| LogNormalError::SolveFailed)?;
    Ok(transpose_owned(&cov_t))
}

fn transpose_owned(matrix: &Mat<f64>) -> Mat<f64> {
    Mat::from_fn(matrix.ncols(), matrix.nrows(), |i, j| matrix[(j, i)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx_to_f64(idx: usize) -> f64 {
        f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
    }

    #[test]
    fn fit_lognormal_smearing_runs() {
        let n = 80;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 });
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 7 == 0 {
                0.0
            } else {
                0.3f64.mul_add(idx_to_f64(i), 2.0)
            }
        });
        let (model, _report) =
            fit_lognormal_smearing(&x, &y, None, LogNormalOptions::default()).expect("fit");
        let pred = model.predict(&x);
        assert_eq!(pred.mean.nrows(), n);
        assert!(model.smearing_factor > 0.0);
    }

    #[test]
    fn lognormal_reports_robust_se_with_clusters() {
        let n = 30;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 });
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 3 == 0 {
                0.0
            } else {
                0.2f64.mul_add(idx_to_f64(i), 1.5)
            }
        });
        let clusters: Vec<u64> = (0..n).map(|i| (i / 5) as u64).collect();
        let options = LogNormalOptions {
            robust_se: true,
            ..LogNormalOptions::default()
        };
        let (_model, report) =
            fit_lognormal_smearing(&x, &y, Some(&clusters), options).expect("fit");
        assert!(report.robust);
        assert!(report.clustered);
        assert!(report.se.is_some());
        assert!(report.cluster_count.unwrap_or(0) > 1);
    }

    #[test]
    fn lognormal_cluster_count_tracks_positive_rows_only() {
        let n = 6;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 });
        let y = Mat::from_fn(n, 1, |i, _| {
            if i < 2 {
                0.0
            } else {
                1.0 + idx_to_f64(i) / 10.0
            }
        });
        let clusters: Vec<u64> = vec![1, 2, 3, 9, 9, 9];
        let options = LogNormalOptions {
            robust_se: true,
            ..LogNormalOptions::default()
        };
        let (_model, report) =
            fit_lognormal_smearing(&x, &y, Some(&clusters), options).expect("fit");
        assert!(report.clustered);
        assert_eq!(report.cluster_count, Some(2));
    }

    #[test]
    fn lognormal_rejects_no_positive_outcomes() {
        let x = Mat::from_fn(
            10,
            2,
            |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 },
        );
        let y = Mat::from_fn(10, 1, |_i, _| 0.0);
        let err = fit_lognormal_smearing(&x, &y, None, LogNormalOptions::default())
            .expect_err("all-zero outcomes should fail");
        assert!(matches!(err, LogNormalError::NoPositiveOutcomes));
    }
}
