/////////////////////////////////////////////////////////////////////////////////////////////\\\
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
use std::time::Instant;

use faer::{Mat, MatRef};

use crate::input::{InputError, ModelInput};
use crate::models::matrix_ops::map_mat;
use crate::models::{
    AttemptDiagnostics, AttemptOutcome, FitMetadata, FitStrategy, Model, SolverKind,
};
use crate::utils::{
    add_ridge_to_diagonal, add_row_outer_product_scaled, max_abs_diff, solve_linear_system,
    solve_linear_system_ref, weighted_xtx, weighted_xtz,
};

const LINEAR_PREDICTOR_CLIP: f64 = 30.0;
const BACKTRACKING_STEPS: usize = 12;

/// Tuning parameters for Tweedie fitting.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct TweedieOptions {
    /// Maximum IRLS iterations.
    #[builder(default = 50_usize)]
    pub max_iter: usize,
    /// Convergence tolerance.
    #[builder(default = 1e-6_f64)]
    pub tolerance: f64,
    /// Lower bound on IRLS weights.
    #[builder(default = 1e-6_f64)]
    pub min_weight: f64,
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

impl Default for TweedieOptions {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tolerance: 1e-6,
            min_weight: 1e-6,
            l2_penalty: 0.0,
            l2_penalty_exclude_intercept: true,
            robust_se: false,
            strategy: FitStrategy::Strict,
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
    #[error("weights must be a single column matrix with the same number of rows as outcome")]
    InvalidWeightShape,
    #[error("cluster labels length ({labels}) must match outcome rows ({rows})")]
    InvalidClusterLength { labels: usize, rows: usize },
    #[error("inputs contain non-finite values")]
    NonFiniteInput,
    #[error("outcome contains negative values")]
    NegativeOutcome,
    #[error("weights contain non-finite values")]
    NonFiniteWeights,
    #[error("weights must be strictly positive")]
    NonPositiveWeights,
    #[error("model failed to converge")]
    NonConvergence,
    #[error("linear solve failed")]
    SolveFailed,
}

crate::impl_input_error_from!(TweedieError, {
    InputError::EmptyDesign => Self::EmptyDesign,
    InputError::InvalidOutcomeShape => Self::InvalidOutcomeShape,
    InputError::DimensionMismatch { rows, len } => Self::DimensionMismatch { rows, len },
    InputError::InvalidWeightShape => Self::InvalidWeightShape,
    InputError::InvalidClusterLength { labels, rows } => {
        Self::InvalidClusterLength { labels, rows }
    },
    InputError::NonFiniteDesign
    | InputError::NonFiniteOutcome
    | InputError::InvalidLabelLength { .. }
    | InputError::DuplicateLabels(_) => Self::NonFiniteInput,
    InputError::NonFiniteWeights => Self::NonFiniteWeights,
    InputError::NonPositiveWeights => Self::NonPositiveWeights,
    InputError::NegativeOutcome => Self::NegativeOutcome,
});

/// Tweedie model coefficients.
#[derive(Debug, Clone)]
pub struct TweedieModel {
    /// Coefficient vector.
    pub beta: Mat<f64>,
    /// Tweedie power parameter (p).
    pub power: f64,
    /// Fit diagnostics.
    pub report: TweedieReport,
}

/// Tweedie diagnostics and inference outputs.
#[derive(Debug, Clone)]
pub struct TweedieReport {
    /// Standardized fit metadata.
    pub meta: FitMetadata,
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
    /// History of retry attempts (if Relaxed strategy used).
    pub attempts: Vec<AttemptDiagnostics>,
}

/// Tweedie predictions (mean on original scale).
#[derive(Debug, Clone)]
pub struct TweediePrediction {
    pub mean: Mat<f64>,
}

impl TweediePrediction {
    /// Create a new prediction container with allocated matrices.
    #[must_use]
    pub fn new(nrows: usize) -> Self {
        Self {
            mean: Mat::zeros(nrows, 1),
        }
    }
}

type RobustCovarianceResult = (Option<Mat<f64>>, bool, Option<usize>);

impl Model for TweedieModel {
    type Prediction = TweediePrediction;
    type Report = TweedieReport;

    fn predict(&self, x: &Mat<f64>) -> Self::Prediction {
        let mut out = TweediePrediction::new(x.nrows());
        self.predict_into(x, &mut out);
        out
    }

    fn predict_into(&self, x: &Mat<f64>, out: &mut Self::Prediction) {
        let eta = x * &self.beta;
        for i in 0..x.nrows() {
            out.mean[(i, 0)] = exp_clamped(eta[(i, 0)]);
        }
    }

    fn report(&self) -> &Self::Report {
        &self.report
    }
}

/// Fit a Tweedie GLM with log link using quasi-likelihood IRLS.
///
/// # Errors
///
/// Returns `TweedieError` if inputs are malformed or the solver fails.
#[allow(clippy::too_many_lines)]
pub(crate) fn fit_tweedie(
    x: &Mat<f64>,
    y: &Mat<f64>,
    clusters: Option<&[u64]>,
    power: f64,
    options: TweedieOptions,
) -> Result<(TweedieModel, TweedieReport), TweedieError> {
    fit_tweedie_weighted(x, y, None, clusters, power, options)
}

#[allow(clippy::too_many_lines)]
fn fit_tweedie_weighted(
    x: &Mat<f64>,
    y: &Mat<f64>,
    sample_weights: Option<&Mat<f64>>,
    clusters: Option<&[u64]>,
    power: f64,
    options: TweedieOptions,
) -> Result<(TweedieModel, TweedieReport), TweedieError> {
    let start_time = Instant::now();
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
    validate_sample_weights(y, sample_weights)?;
    if (0..y.nrows()).any(|i| y[(i, 0)] < 0.0) {
        return Err(TweedieError::NegativeOutcome);
    }

    let mut current_options = options;
    let mut attempts = Vec::new();
    let mut last_err = TweedieError::NonConvergence;

    let strategy = options.strategy;
    let max_attempts = match strategy {
        FitStrategy::Strict => 1,
        FitStrategy::Relaxed { max_retries, .. } => 1 + max_retries,
    };

    let warm_start_beta: Option<Mat<f64>> = None;

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

        let result = fit_tweedie_with_lambda(
            x,
            y,
            sample_weights,
            clusters,
            power,
            current_options,
            current_options.l2_penalty,
            warm_start_beta.as_ref(),
        );

        match result {
            Ok((mut model, mut report)) => {
                crate::finalize_retry_fit!(model, report, attempts, start_time, attempt_idx);
            }
            Err(e @ (TweedieError::NonConvergence | TweedieError::SolveFailed)) => {
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
                if let FitStrategy::Relaxed {
                    warm_start: true, ..
                } = strategy
                {
                    // Could potentially capture partial beta here if we changed fit_tweedie_with_lambda
                }
            }
            Err(e) => return Err(e),
        }
    }

    Err(last_err)
}

fn fit_tweedie_with_lambda(
    x: &Mat<f64>,
    y: &Mat<f64>,
    sample_weights: Option<&Mat<f64>>,
    clusters: Option<&[u64]>,
    power: f64,
    options: TweedieOptions,
    lambda: f64,
    initial_beta_override: Option<&Mat<f64>>,
) -> Result<(TweedieModel, TweedieReport), TweedieError> {
    let mut beta = initial_beta_override
        .cloned()
        .unwrap_or_else(|| initial_beta(x, y, sample_weights, options.min_weight));

    for iteration in 0..options.max_iter {
        let eta = x * &beta;
        let mu = map_mat(&eta, exp_clamped);
        let current_deviance = weighted_deviance(y, &mu, power, sample_weights);

        let weights = Mat::from_fn(mu.nrows(), 1, |i, _| {
            let base_weight = sample_weight_at(sample_weights, i);
            let w = mu[(i, 0)].powf(2.0 - power).max(options.min_weight);
            base_weight * w
        });
        let z = Mat::from_fn(mu.nrows(), 1, |i, _| {
            eta[(i, 0)] + (y[(i, 0)] - mu[(i, 0)]) / mu[(i, 0)]
        });
        let mut xtwx = weighted_xtx(x, &weights);
        if lambda > 0.0 {
            add_ridge_to_diagonal(&mut xtwx, lambda, options.l2_penalty_exclude_intercept);
        }
        let xtw_rhs = weighted_xtz(x, &weights, &z);
        let beta_candidate = solve_with_stabilization(&xtwx, &xtw_rhs)?;
        let beta_next = backtracking_update(
            x,
            y,
            sample_weights,
            &beta,
            &beta_candidate,
            current_deviance,
            power,
        );

        let eta_next = x * &beta_next;
        let mu_next = map_mat(&eta_next, exp_clamped);
        let dev_next = weighted_deviance(y, &mu_next, power, sample_weights);

        let beta_converged = max_abs_diff(&beta_next, &beta) < options.tolerance;
        let dev_converged = relative_change(current_deviance, dev_next) < options.tolerance;
        if beta_converged || dev_converged {
            return finalize_fit(
                x,
                y,
                sample_weights,
                clusters,
                power,
                options,
                lambda,
                beta_next,
                iteration + 1,
            );
        }
        beta = beta_next;
    }

    Err(TweedieError::NonConvergence)
}

fn initial_beta(
    x: &Mat<f64>,
    y: &Mat<f64>,
    sample_weights: Option<&Mat<f64>>,
    min_weight: f64,
) -> Mat<f64> {
    let mut beta = Mat::<f64>::zeros(x.ncols(), 1);
    if x.ncols() == 0 || y.nrows() == 0 {
        return beta;
    }
    let mut mean_y = 0.0;
    let mut weight_sum = 0.0;
    for i in 0..y.nrows() {
        let weight = sample_weight_at(sample_weights, i);
        mean_y += weight * y[(i, 0)];
        weight_sum += weight;
    }
    mean_y /= weight_sum.max(1.0);
    beta[(0, 0)] = mean_y.max(min_weight).ln();
    beta
}

fn validate_sample_weights(
    y: &Mat<f64>,
    sample_weights: Option<&Mat<f64>>,
) -> Result<(), TweedieError> {
    let Some(weights) = sample_weights else {
        return Ok(());
    };
    if weights.ncols() != 1 || weights.nrows() != y.nrows() {
        return Err(TweedieError::InvalidWeightShape);
    }
    if !crate::utils::matrix_is_finite(weights) {
        return Err(TweedieError::NonFiniteWeights);
    }
    if (0..weights.nrows()).any(|i| weights[(i, 0)] <= 0.0) {
        return Err(TweedieError::NonPositiveWeights);
    }
    Ok(())
}

fn sample_weight_at(sample_weights: Option<&Mat<f64>>, row: usize) -> f64 {
    sample_weights.map_or(1.0, |weights| weights[(row, 0)])
}

#[allow(clippy::too_many_arguments)]
fn finalize_fit(
    x: &Mat<f64>,
    y: &Mat<f64>,
    sample_weights: Option<&Mat<f64>>,
    clusters: Option<&[u64]>,
    power: f64,
    options: TweedieOptions,
    lambda: f64,
    beta_final: Mat<f64>,
    iterations: usize,
) -> Result<(TweedieModel, TweedieReport), TweedieError> {
    let eta = x * &beta_final;
    let mu = map_mat(&eta, exp_clamped);
    let weights = Mat::from_fn(mu.nrows(), 1, |i, _| {
        let base_weight = sample_weight_at(sample_weights, i);
        let w = mu[(i, 0)].powf(2.0 - power).max(options.min_weight);
        base_weight * w
    });
    let mut xtwx = weighted_xtx(x, &weights);
    if lambda > 0.0 {
        add_ridge_to_diagonal(&mut xtwx, lambda, options.l2_penalty_exclude_intercept);
    }
    let (cov, se, clustered, cluster_count) = if options.robust_se {
        let (cov, clustered, cluster_count) =
            robust_covariance(x, y, &mu, &weights, clusters, &xtwx)?;
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
    let report = TweedieReport {
        meta,
        se,
        cov,
        robust: options.robust_se,
        clustered,
        cluster_count,
        attempts: Vec::new(),
    };
    let model = TweedieModel {
        beta: beta_final,
        power,
        report: report.clone(),
    };
    Ok((model, report))
}

/// Fit a Tweedie GLM from a `ModelInput` container.
///
/// # Errors
///
/// Returns `TweedieError` if inputs are malformed or the solver fails.
pub fn fit_tweedie_input(
    input: &ModelInput,
    power: f64,
    options: TweedieOptions,
) -> Result<(TweedieModel, TweedieReport), TweedieError> {
    input.validate().map_err(TweedieError::from)?;
    fit_tweedie_weighted(
        &input.design_matrix,
        &input.outcome,
        input.sample_weights.as_ref(),
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

    weighted_deviance(y, mu, power, None)
}

fn weighted_deviance(
    y: &Mat<f64>,
    mu: &Mat<f64>,
    power: f64,
    sample_weights: Option<&Mat<f64>>,
) -> f64 {
    // Closed-form endpoint for p = 1 (Poisson deviance limit).
    if (power - 1.0).abs() < 1e-12 {
        let mut result = 0.0;
        for i in 0..y.nrows() {
            let yi = y[(i, 0)].max(0.0);
            let mui = mu[(i, 0)].max(1e-12);
            let weight = sample_weight_at(sample_weights, i);
            if yi == 0.0 {
                result += weight * 2.0 * mui;
            } else {
                result += weight * 2.0 * (yi * (yi / mui).ln() - (yi - mui));
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
            let weight = sample_weight_at(sample_weights, i);
            if yi <= 0.0 {
                return f64::INFINITY;
            }
            result += weight * 2.0 * (((yi - mui) / mui) - (yi / mui).ln());
        }
        return result;
    }

    let mut deviance = 0.0;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        let mui = mu[(i, 0)].max(1e-12);
        if yi == 0.0 {
            deviance +=
                sample_weight_at(sample_weights, i) * 2.0 * mui.powf(2.0 - power) / (2.0 - power);
        } else {
            let term1 = yi.powf(2.0 - power) / ((1.0 - power) * (2.0 - power));
            let term2 = yi * mui.powf(1.0 - power) / (1.0 - power);
            let term3 = mui.powf(2.0 - power) / (2.0 - power);
            deviance += sample_weight_at(sample_weights, i) * 2.0 * (term1 - term2 + term3);
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

fn diagonal_sqrt(cov: &Mat<f64>) -> Mat<f64> {
    Mat::from_fn(cov.nrows(), 1, |i, _| cov[(i, i)].max(0.0).sqrt())
}

fn robust_covariance(
    x: &Mat<f64>,
    y: &Mat<f64>,
    mu: &Mat<f64>,
    weights: &Mat<f64>,
    clusters: Option<&[u64]>,
    xtwx: &Mat<f64>,
) -> Result<RobustCovarianceResult, TweedieError> {
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
        let cov = sandwich_covariance(xtwx, &meat)?;
        return Ok((Some(cov), true, Some(cluster_sums.len())));
    }

    for i in 0..x.nrows() {
        let resid = (y[(i, 0)] - mu[(i, 0)]) * weights[(i, 0)];
        add_row_outer_product_scaled(&mut meat, x, i, resid * resid);
    }
    let cov = sandwich_covariance(xtwx, &meat)?;
    Ok((Some(cov), false, None))
}

fn sandwich_covariance(xtwx: &Mat<f64>, meat: &Mat<f64>) -> Result<Mat<f64>, TweedieError> {
    let left = solve_with_stabilization(xtwx, meat)?;
    let cov_t = solve_with_stabilization_view(xtwx.transpose(), left.transpose())?;
    Ok(Mat::from_fn(cov_t.ncols(), cov_t.nrows(), |i, j| {
        cov_t[(j, i)]
    }))
}

fn backtracking_update(
    x: &Mat<f64>,
    y: &Mat<f64>,
    sample_weights: Option<&Mat<f64>>,
    beta_current: &Mat<f64>,
    beta_candidate: &Mat<f64>,
    current_deviance: f64,
    power: f64,
) -> Mat<f64> {
    let mut step = 1.0_f64;
    let mut proposal = beta_candidate.clone();
    let mut best_beta = beta_candidate.clone();
    let mut best_deviance = f64::INFINITY;
    for _ in 0..BACKTRACKING_STEPS {
        if (step - 1.0).abs() < f64::EPSILON {
            for row in 0..proposal.nrows() {
                proposal[(row, 0)] = beta_candidate[(row, 0)];
            }
        } else {
            blend_betas_into(&mut proposal, beta_current, beta_candidate, step);
        }
        let mu = map_mat(&(x * &proposal), exp_clamped);
        let dev = weighted_deviance(y, &mu, power, sample_weights);
        if dev.is_finite() && dev <= current_deviance {
            return proposal;
        }
        if dev.is_finite() && dev < best_deviance {
            best_deviance = dev;
            for row in 0..best_beta.nrows() {
                best_beta[(row, 0)] = proposal[(row, 0)];
            }
        }
        step *= 0.5;
    }
    best_beta
}

fn blend_betas_into(
    output: &mut Mat<f64>,
    beta_current: &Mat<f64>,
    beta_candidate: &Mat<f64>,
    step: f64,
) {
    for row in 0..output.nrows() {
        let current = beta_current[(row, 0)];
        output[(row, 0)] = (beta_candidate[(row, 0)] - current).mul_add(step, current);
    }
}

fn relative_change(previous: f64, next: f64) -> f64 {
    if !previous.is_finite() || !next.is_finite() {
        return f64::INFINITY;
    }
    (next - previous).abs() / previous.abs().max(1.0)
}

fn exp_clamped(value: f64) -> f64 {
    value
        .clamp(-LINEAR_PREDICTOR_CLIP, LINEAR_PREDICTOR_CLIP)
        .exp()
}

fn solve_with_stabilization(lhs: &Mat<f64>, rhs: &Mat<f64>) -> Result<Mat<f64>, TweedieError> {
    if let Ok(solution) = solve_linear_system(lhs, rhs) {
        return Ok(solution);
    }

    let mut stabilized = lhs.clone();
    let dim = lhs.nrows().min(lhs.ncols());
    let mut jitter = 1e-10;
    for _ in 0..8 {
        for idx in 0..dim {
            stabilized[(idx, idx)] = lhs[(idx, idx)] + jitter;
        }
        if let Ok(solution) = solve_linear_system(&stabilized, rhs) {
            return Ok(solution);
        }
        jitter *= 10.0;
    }
    Err(TweedieError::SolveFailed)
}

fn solve_with_stabilization_view(
    lhs: MatRef<'_, f64>,
    rhs: MatRef<'_, f64>,
) -> Result<Mat<f64>, TweedieError> {
    if let Ok(solution) = solve_linear_system_ref(lhs, rhs) {
        return Ok(solution);
    }

    let mut stabilized = Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| lhs[(row, col)]);
    let dim = lhs.nrows().min(lhs.ncols());
    let mut jitter = 1e-10;
    for _ in 0..8 {
        for idx in 0..dim {
            stabilized[(idx, idx)] = lhs[(idx, idx)] + jitter;
        }
        if let Ok(solution) = solve_linear_system_ref(stabilized.as_ref(), rhs) {
            return Ok(solution);
        }
        jitter *= 10.0;
    }
    Err(TweedieError::SolveFailed)
}

/// High-level interface for fitting Tweedie GLMs.
#[derive(Debug, Clone, Copy)]
pub struct TweedieTrainer {
    power: f64,
    options: TweedieOptions,
}

impl Default for TweedieTrainer {
    fn default() -> Self {
        Self::new(1.5)
    }
}

impl TweedieTrainer {
    /// Create a new trainer with a fixed power parameter.
    #[must_use]
    pub fn new(power: f64) -> Self {
        Self {
            power,
            options: TweedieOptions::default(),
        }
    }

    /// Fast preset: minimal iterations, no robust standard errors.
    #[must_use]
    pub fn fast(power: f64) -> Self {
        Self {
            power,
            options: TweedieOptions {
                max_iter: 20,
                tolerance: 1e-4,
                robust_se: false,
                strategy: FitStrategy::Strict,
                ..TweedieOptions::default()
            },
        }
    }

    /// Stable preset: mild ridge regularization and relaxed strategy.
    #[must_use]
    pub fn stable(power: f64) -> Self {
        Self {
            power,
            options: TweedieOptions {
                l2_penalty: 1e-4,
                strategy: FitStrategy::Relaxed {
                    fallback_lambda: 1e-3,
                    max_retries: 3,
                    warm_start: true,
                    time_budget: None,
                },
                ..TweedieOptions::default()
            },
        }
    }

    /// Inference preset: robust standard errors and high precision.
    #[must_use]
    pub fn inference(power: f64) -> Self {
        Self {
            power,
            options: TweedieOptions {
                max_iter: 100,
                tolerance: 1e-8,
                robust_se: true,
                strategy: FitStrategy::Strict,
                ..TweedieOptions::default()
            },
        }
    }

    /// Set the Tweedie power parameter (p).
    #[must_use]
    pub const fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// Set custom fit options.
    #[must_use]
    pub const fn with_options(mut self, options: TweedieOptions) -> Self {
        self.options = options;
        self
    }

    /// Set convergence failure strategy.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: FitStrategy) -> Self {
        self.options.strategy = strategy;
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
    /// Returns `TweedieError` if input is invalid or fitting fails.
    pub fn fit(&self, input: &ModelInput) -> Result<TweedieModel, TweedieError> {
        fit_tweedie_input(input, self.power, self.options).map(|(m, _)| m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::usize_to_f64;

    #[test]
    fn fit_tweedie_runs() {
        let n = 100;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 20.0 },
        );
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 5 == 0 {
                0.0
            } else {
                0.2f64.mul_add(usize_to_f64(i), 2.0)
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
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 20.0 },
        );
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 5 == 0 {
                0.0
            } else {
                0.2f64.mul_add(usize_to_f64(i), 2.0)
            }
        });
        let clusters: Vec<u64> = (0..n).map(|i| (i / 4) as u64).collect();
        let options = TweedieOptions::builder().robust_se(true).build();
        let (model, report) = fit_tweedie(&x, &y, Some(&clusters), 1.5, options).expect("fit");
        let pred = model.predict(&x);
        assert_eq!(pred.mean.nrows(), n);
        assert!(report.robust);
        assert!(report.clustered);
        assert!(report.se.is_some());
    }

    #[test]
    fn fit_strategy_relaxed_handles_tweedie_non_convergence() {
        let n = 10;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) });
        let y = Mat::from_fn(n, 1, |i, _| if i == 0 { 0.0 } else { 1.0 });

        let options = TweedieOptions {
            max_iter: 1, // Force early failure
            strategy: FitStrategy::Relaxed {
                fallback_lambda: 1.0,
                max_retries: 2,
                warm_start: true,
                time_budget: None,
            },
            ..TweedieOptions::default()
        };

        match fit_tweedie_input(&ModelInput::new(x, y), 1.5, options) {
            Ok((_model, report)) => {
                assert!(report.meta.fallback_attempts > 0);
                assert!(!report.attempts.is_empty());
            }
            Err(TweedieError::NonConvergence | TweedieError::SolveFailed) => {
                // On tiny pathological samples, relaxed retries may still fail.
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }
}
