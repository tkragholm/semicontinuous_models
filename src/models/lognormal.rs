/////////////////////////////////////////////////////////////////////////////////////////////\\\
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
use std::time::Instant;

use faer::Mat;

use crate::models::matrix_ops::{map_mat, select_rows, select_values};
use crate::models::{
    AttemptDiagnostics, AttemptOutcome, FitMetadata, FitStrategy, Model, SolverKind,
};
use thiserror::Error;

use crate::input::{InputError, ModelInput};
use crate::utils::{
    add_ridge_to_diagonal, add_row_outer_product_scaled, max_abs_diff, mean_column,
    solve_linear_system, solve_linear_system_ref,
};

/// Tuning parameters for log-normal fitting.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct LogNormalOptions {
    /// Maximum number of iterations (usually 1 for OLS).
    #[builder(default = 1_usize)]
    pub max_iter: usize,
    /// Convergence tolerance.
    #[builder(default = 1e-12_f64)]
    pub tolerance: f64,
    /// L2 (ridge) penalty strength.
    #[builder(default = 0.0_f64)]
    pub l2_penalty: f64,
    /// If true, do not penalize the first column (intercept).
    #[builder(default = true)]
    pub l2_penalty_exclude_intercept: bool,
    /// If true, compute robust (sandwich) covariance.
    #[builder(default = false)]
    pub robust_se: bool,
    /// Strategy for handling non-convergence.
    #[builder(default = FitStrategy::Strict)]
    pub strategy: FitStrategy,
}

impl Default for LogNormalOptions {
    fn default() -> Self {
        Self {
            max_iter: 1,
            tolerance: 1e-12,
            l2_penalty: 0.0,
            l2_penalty_exclude_intercept: true,
            robust_se: false,
            strategy: FitStrategy::Strict,
        }
    }
}

impl LogNormalOptions {
    /// Stable defaults for noisy observational data.
    ///
    /// Uses a mild L2 penalty to make linear solves more robust in near-singular designs.
    #[must_use]
    pub fn stable_defaults() -> Self {
        Self::builder()
            .l2_penalty(1e-4)
            .strategy(FitStrategy::Relaxed {
                fallback_lambda: 1e-3,
                max_retries: 3,
                warm_start: false,
                time_budget: None,
            })
            .build()
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
    #[error("model failed to converge")]
    NonConvergence,
}

crate::impl_input_error_from!(LogNormalError, {
    InputError::EmptyDesign => Self::EmptyDesign,
    InputError::InvalidOutcomeShape => Self::InvalidOutcomeShape,
    InputError::DimensionMismatch { rows, len } => Self::DimensionMismatch { rows, len },
    InputError::InvalidClusterLength { labels, rows } => {
        Self::InvalidClusterLength { labels, rows }
    },
    InputError::NonFiniteDesign
    | InputError::NonFiniteOutcome
    | InputError::InvalidWeightShape
    | InputError::NonFiniteWeights
    | InputError::NonPositiveWeights
    | InputError::InvalidLabelLength { .. }
    | InputError::DuplicateLabels(_) => Self::NonFiniteInput,
    InputError::NegativeOutcome => Self::NegativeOutcome,
});

/// Log-normal model coefficients and smearing factor.
#[derive(Debug, Clone)]
pub struct LogNormalModel {
    /// Coefficient vector for log-outcomes.
    pub beta: Mat<f64>,
    /// Smearing factor for retransformation.
    pub smearing_factor: f64,
    /// Fit diagnostics.
    pub report: LogNormalReport,
}

/// Model diagnostics and inference outputs.
#[derive(Debug, Clone)]
pub struct LogNormalReport {
    /// Standardized fit metadata.
    pub meta: FitMetadata,
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
    /// History of retry attempts (if Relaxed strategy used).
    pub attempts: Vec<AttemptDiagnostics>,
}

/// Log-normal predictions on the original scale.
#[derive(Debug, Clone)]
pub struct LogNormalPrediction {
    pub mean: Mat<f64>,
}

impl LogNormalPrediction {
    /// Create a new prediction container with allocated matrices.
    #[must_use]
    pub fn new(nrows: usize) -> Self {
        Self {
            mean: Mat::zeros(nrows, 1),
        }
    }
}

type RobustCovarianceResult = (Option<Mat<f64>>, bool, Option<usize>);

impl Model for LogNormalModel {
    type Prediction = LogNormalPrediction;
    type Report = LogNormalReport;

    fn predict(&self, x: &Mat<f64>) -> Self::Prediction {
        let mut out = LogNormalPrediction::new(x.nrows());
        self.predict_into(x, &mut out);
        out
    }

    fn predict_into(&self, x: &Mat<f64>, out: &mut Self::Prediction) {
        let eta = x * &self.beta;
        for i in 0..x.nrows() {
            out.mean[(i, 0)] = self.smearing_factor * eta[(i, 0)].exp();
        }
    }

    fn report(&self) -> &Self::Report {
        &self.report
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
            sumsq = resid.mul_add(resid, sumsq);
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
    let start_time = Instant::now();
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

    let mut current_options = options;
    let mut attempts = Vec::new();
    let mut last_err = LogNormalError::SolveFailed;

    let strategy = options.strategy;
    let max_attempts = match strategy {
        FitStrategy::Strict => 1,
        FitStrategy::Relaxed { max_retries, .. } => 1 + max_retries,
    };

    for attempt_idx in 0..max_attempts {
        if attempt_idx > 0
            && let FitStrategy::Relaxed {
                fallback_lambda,
                time_budget,
                ..
            } = strategy
        {
            if let Some(budget) = time_budget
                && start_time.elapsed() >= budget
            {
                attempts.push(AttemptDiagnostics {
                    attempt: attempt_idx,
                    lambda_used: 0.0,
                    meta: FitMetadata::default(),
                    outcome: AttemptOutcome::TimedOut,
                });
                break;
            }

            // Incremental regularization fallback
            let scale = 10.0f64.powi(i32::try_from(attempt_idx - 1).unwrap_or(0));
            current_options.l2_penalty =
                fallback_lambda.mul_add(scale, options.l2_penalty).max(1e-8);
        }

        let result = fit_lognormal_with_lambda(x, y, clusters, current_options);

        match result {
            Ok((mut model, mut report)) => {
                crate::finalize_retry_fit!(model, report, attempts, start_time, attempt_idx);
            }
            Err(e @ (LogNormalError::SolveFailed | LogNormalError::NonConvergence)) => {
                attempts.push(AttemptDiagnostics {
                    attempt: attempt_idx,
                    lambda_used: current_options.l2_penalty,
                    meta: FitMetadata {
                        converged: false,
                        execution_time: start_time.elapsed(),
                        ..FitMetadata::default()
                    },
                    outcome: AttemptOutcome::Diverged,
                });
                last_err = e;
            }
            Err(e) => return Err(e),
        }
    }

    Err(last_err)
}

fn fit_lognormal_with_lambda(
    x: &Mat<f64>,
    y: &Mat<f64>,
    clusters: Option<&[u64]>,
    options: LogNormalOptions,
) -> Result<(LogNormalModel, LogNormalReport), LogNormalError> {
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
    let mut iterations = 0;

    for iteration in 0..options.max_iter {
        iterations = iteration + 1;
        let mut xtx = x_pos.transpose() * &x_pos;
        add_ridge_to_diagonal(&mut xtx, lambda, options.l2_penalty_exclude_intercept);
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
        let mut xtx = x_pos.transpose() * &x_pos;
        add_ridge_to_diagonal(&mut xtx, lambda, options.l2_penalty_exclude_intercept);
        let (cov, clustered, cluster_count) =
            robust_covariance(&x_pos, &residuals, cluster_ids_pos.as_deref(), &xtx)?;
        let se = cov.as_ref().map(diagonal_sqrt);
        (cov, se, clustered, cluster_count)
    } else {
        (None, None, false, None)
    };

    let meta = FitMetadata {
        iterations,
        converged: true,
        solver: SolverKind::Irls,
        ..FitMetadata::default()
    };
    let report = LogNormalReport {
        meta,
        se,
        cov,
        robust: options.robust_se,
        clustered,
        cluster_count,
        attempts: Vec::new(),
    };
    let model = LogNormalModel {
        beta,
        smearing_factor: smearing,
        report: report.clone(),
    };

    Ok((model, report))
}

/// Fit a log-normal model with smearing from a `ModelInput` container.
///
/// # Errors
///
/// Returns `LogNormalError` if inputs are malformed or no positive outcomes exist.
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
                entry[j] = resid.mul_add(x[(i, j)], entry[j]);
            }
        }
        for sum in cluster_sums.values() {
            for j in 0..p {
                for k in 0..p {
                    meat[(j, k)] = sum[j].mul_add(sum[k], meat[(j, k)]);
                }
            }
        }
        let cov = sandwich_covariance(xtx, &meat)?;
        return Ok((Some(cov), true, Some(cluster_sums.len())));
    }

    for i in 0..x.nrows() {
        let resid = residuals[(i, 0)];
        add_row_outer_product_scaled(&mut meat, x, i, resid * resid);
    }
    let cov = sandwich_covariance(xtx, &meat)?;
    Ok((Some(cov), false, None))
}

fn sandwich_covariance(xtx: &Mat<f64>, meat: &Mat<f64>) -> Result<Mat<f64>, LogNormalError> {
    let left = solve_linear_system(xtx, meat).map_err(|_| LogNormalError::SolveFailed)?;
    let cov_t = solve_linear_system_ref(xtx.transpose(), left.transpose())
        .map_err(|_| LogNormalError::SolveFailed)?;
    Ok(Mat::from_fn(cov_t.ncols(), cov_t.nrows(), |i, j| {
        cov_t[(j, i)]
    }))
}

/// High-level interface for fitting log-normal models.
#[derive(Debug, Clone, Copy)]
pub struct LogNormalTrainer {
    options: LogNormalOptions,
}

impl Default for LogNormalTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl LogNormalTrainer {
    /// Create a new trainer with default options.
    #[must_use]
    pub fn new() -> Self {
        Self {
            options: LogNormalOptions::default(),
        }
    }

    /// Fast preset: minimal precision, no robust standard errors.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            options: LogNormalOptions {
                max_iter: 1,
                tolerance: 1e-4,
                robust_se: false,
                strategy: FitStrategy::Strict,
                ..LogNormalOptions::default()
            },
        }
    }

    /// Stable preset: mild ridge regularization and relaxed strategy.
    #[must_use]
    pub fn stable() -> Self {
        Self {
            options: LogNormalOptions::stable_defaults(),
        }
    }

    /// Inference preset: robust standard errors and high precision.
    #[must_use]
    pub fn inference() -> Self {
        Self {
            options: LogNormalOptions {
                max_iter: 1,
                tolerance: 1e-12,
                robust_se: true,
                strategy: FitStrategy::Strict,
                ..LogNormalOptions::default()
            },
        }
    }

    /// Set custom fit options.
    #[must_use]
    pub const fn with_options(mut self, options: LogNormalOptions) -> Self {
        self.options = options;
        self
    }

    /// Set convergence failure strategy.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: FitStrategy) -> Self {
        self.options.strategy = strategy;
        self
    }

    /// Set convergence tolerance.
    #[must_use]
    pub const fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.options.tolerance = tolerance;
        self
    }

    /// Set L2 (ridge) penalty strength.
    #[must_use]
    pub const fn with_l2_penalty(mut self, l2_penalty: f64) -> Self {
        self.options.l2_penalty = l2_penalty;
        self
    }

    /// Enable or disable robust standard errors.
    #[must_use]
    pub const fn with_robust_se(mut self, enabled: bool) -> Self {
        self.options.robust_se = enabled;
        self
    }

    /// Fit the model to the provided input.
    ///
    /// # Errors
    ///
    /// Returns `LogNormalError` if input is invalid or fitting fails.
    pub fn fit(&self, input: &ModelInput) -> Result<LogNormalModel, LogNormalError> {
        fit_lognormal_smearing_input(input, self.options).map(|(m, _)| m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::usize_to_f64;

    #[test]
    fn fit_lognormal_smearing_runs() {
        let n = 80;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 },
        );
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 7 == 0 {
                0.0
            } else {
                0.3f64.mul_add(usize_to_f64(i), 2.0)
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
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 },
        );
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 3 == 0 {
                0.0
            } else {
                0.2f64.mul_add(usize_to_f64(i), 1.5)
            }
        });
        let clusters: Vec<u64> = (0..n).map(|i| (i / 5) as u64).collect();
        let options = LogNormalOptions::builder().robust_se(true).build();
        let (_model, report) =
            fit_lognormal_smearing(&x, &y, Some(&clusters), options).expect("fit");
        assert!(report.robust);
        assert!(report.clustered);
        assert!(report.se.is_some());
        assert!(report.cluster_count.unwrap_or(0) > 1);
    }

    #[test]
    fn fit_strategy_relaxed_handles_lognormal_solve_failure() {
        let n = 5;
        // Near-singular design
        let x = Mat::from_fn(n, 2, |_, _| 1.0);
        let y = Mat::from_fn(n, 1, |_, _| 10.0);

        let options = LogNormalOptions {
            l2_penalty: 0.0,
            strategy: FitStrategy::Relaxed {
                fallback_lambda: 1.0,
                max_retries: 1,
                warm_start: false,
                time_budget: None,
            },
            ..LogNormalOptions::default()
        };

        // This might still solve depending on floating point, but ridge should make it safer.
        let (_model, report) =
            fit_lognormal_smearing_input(&ModelInput::new(x, y), options).expect("fit");
        // If it succeeded without retry, fallback_attempts is 0.
        // If it failed and retried with lambda, fallback_attempts is 1.
        assert!(report.meta.converged);
    }
}
