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
use crate::models::matrix_ops::{
    center_beta, center_columns, map_mat, max_abs_linear_predictor, uncenter_beta,
    weighted_column_means,
};
use crate::models::{
    AttemptDiagnostics, AttemptOutcome, FitMetadata, FitStrategy, Model, SolverKind,
};
use crate::utils::{
    CachedFactor, add_outer_product_scaled, add_ridge_to_diagonal, add_row_outer_product_scaled,
    constant_irls_weights, matvec_into, max_abs_diff, solve_linear_system, solve_linear_system_ref,
    weighted_xtx, weighted_xtz_with_buffer,
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
    // Raw entry point: validate the data (finiteness / negativity) inside the fit.
    fit_tweedie_weighted(x, y, None, clusters, power, options, false, None)
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn fit_tweedie_weighted(
    x: &Mat<f64>,
    y: &Mat<f64>,
    sample_weights: Option<&Mat<f64>>,
    clusters: Option<&[u64]>,
    power: f64,
    options: TweedieOptions,
    // When the caller already validated the data via `ModelInput::validate` (e.g.
    // `fit_tweedie_input`, and every bootstrap replicate through it), skip the two
    // O(n·p) finiteness / negativity scans here — a row-gather of finite, non-negative
    // rows is still finite and non-negative. The cheap O(1) shape checks always run.
    assume_validated: bool,
    // Warm-start coefficients for the IRLS. A bootstrap replicate is a perturbation of
    // the full sample, so seeding from the full-sample fit converges in far fewer
    // iterations to the SAME optimum. `None` cold-starts from the intercept-only model.
    initial_beta: Option<&Mat<f64>>,
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
    if !assume_validated
        && (!crate::utils::matrix_is_finite(x) || !crate::utils::matrix_is_finite(y))
    {
        return Err(TweedieError::NonFiniteInput);
    }
    validate_sample_weights(y, sample_weights)?;
    if !assume_validated && (0..y.nrows()).any(|i| y[(i, 0)] < 0.0) {
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

        let result = fit_tweedie_with_lambda(TweedieFitRequest {
            x,
            y,
            sample_weights,
            clusters,
            power,
            options: current_options,
            lambda: current_options.l2_penalty,
            initial_beta_override: initial_beta,
        });

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

#[derive(Clone, Copy)]
struct TweedieFitRequest<'a> {
    x: &'a Mat<f64>,
    y: &'a Mat<f64>,
    sample_weights: Option<&'a Mat<f64>>,
    clusters: Option<&'a [u64]>,
    power: f64,
    options: TweedieOptions,
    lambda: f64,
    initial_beta_override: Option<&'a Mat<f64>>,
}

fn fit_tweedie_with_lambda(
    request: TweedieFitRequest<'_>,
) -> Result<(TweedieModel, TweedieReport), TweedieError> {
    let TweedieFitRequest {
        x,
        y,
        sample_weights,
        clusters,
        power,
        options,
        lambda,
        initial_beta_override,
    } = request;

    // Center the covariate columns for numerical conditioning (see the module note in
    // `matrix_ops`). The IRLS below runs entirely on the centered design `cx`, so the
    // intercept stays near `ln(mean y)` and η stays in the data-supported range instead of
    // drifting into the `exp_clamped` saturation ceiling. `col_means` is `None` when the
    // design has no usable intercept in column 0, in which case `cx == x` (no centering).
    // The converged coefficients are un-centered back to the raw-x scale before
    // `finalize_fit`, which then computes SE/cov (and everything downstream — predict, the
    // marginal standardisation) on the original design exactly as before.
    let col_means = weighted_column_means(x, sample_weights);
    let centered_storage;
    let cx: &Mat<f64> = match &col_means {
        Some(means) => {
            centered_storage = center_columns(x, means);
            &centered_storage
        }
        None => x,
    };

    let mut beta = match (initial_beta_override, &col_means) {
        // Warm-start coefficients arrive on the raw-x scale; map them onto the centered
        // scale so IRLS resumes from the same fitted model.
        (Some(init), Some(means)) => center_beta(init, means),
        (Some(init), None) => init.clone(),
        (None, _) => initial_beta(cx, y, sample_weights, options.min_weight),
    };

    // For power == 2 (gamma) the Fisher-scoring weight `μ^(2-power) == 1`, so the
    // information matrix `X'WX` (W = the fixed sample weights) is invariant across
    // iterations. Build and factor it once here; each iteration then only recomputes
    // the working-response RHS and back-substitutes — removing the O(n·p²) Gram build
    // and O(p³) factorization from every pass but the first. Exact, not an
    // approximation: the cached factor is the true information matrix at every iterate,
    // so the β-path is identical to rebuilding it each iteration.
    let constant_weight = if (2.0 - power).abs() < 1e-12 {
        let weights = constant_irls_weights(sample_weights, cx.nrows());
        let mut xtwx = weighted_xtx(cx, &weights);
        if lambda > 0.0 {
            add_ridge_to_diagonal(&mut xtwx, lambda, options.l2_penalty_exclude_intercept);
        }
        Some((weights, CachedFactor::factor(&xtwx)))
    } else {
        None
    };

    // Column buffers reused across iterations so the per-replicate IRLS does not churn
    // the allocator (the bootstrap runs thousands of these fits in parallel). `eta`/`mu`/
    // `z` are the linear predictor, mean and working response; `search_*` are line-search
    // scratch; `weight_buffer` holds the per-iteration weight on the variable-weight path.
    let n = cx.nrows();
    let mut eta = Mat::<f64>::zeros(n, 1);
    let mut mu = Mat::<f64>::zeros(n, 1);
    let mut z = Mat::<f64>::zeros(n, 1);
    let mut xtz_buffer: Vec<f64> = Vec::new();
    let mut search_eta = Mat::<f64>::zeros(n, 1);
    let mut search_mu = Mat::<f64>::zeros(n, 1);
    let mut weight_buffer = Mat::<f64>::zeros(n, 1);

    for iteration in 0..options.max_iter {
        matvec_into(&mut eta, cx, &beta);
        for i in 0..n {
            mu[(i, 0)] = exp_clamped(eta[(i, 0)]);
        }
        let current_deviance = weighted_deviance(y, &mu, power, sample_weights);

        // Working response `z = η + (y - μ)/μ`; independent of the IRLS weights.
        for i in 0..n {
            z[(i, 0)] = eta[(i, 0)] + (y[(i, 0)] - mu[(i, 0)]) / mu[(i, 0)];
        }

        let beta_candidate = if let Some((weights, factor)) = &constant_weight {
            // Constant-weight path: reuse the cached factor; only the RHS changes.
            let xtw_rhs = weighted_xtz_with_buffer(cx, weights, &z, &mut xtz_buffer);
            match factor.solve(xtw_rhs.as_ref()) {
                Ok(candidate) => candidate,
                Err(_) => {
                    // Pathological RHS: stabilized solve on the (constant) matrix.
                    let mut xtwx = weighted_xtx(cx, weights);
                    if lambda > 0.0 {
                        add_ridge_to_diagonal(
                            &mut xtwx,
                            lambda,
                            options.l2_penalty_exclude_intercept,
                        );
                    }
                    solve_with_stabilization(&xtwx, &xtw_rhs)?
                }
            }
        } else {
            // Variable-weight path (1 < power < 2): the weight depends on μ, so the
            // information matrix must be rebuilt and factored each iteration.
            for i in 0..n {
                let base_weight = sample_weight_at(sample_weights, i);
                let w = mu[(i, 0)].powf(2.0 - power).max(options.min_weight);
                weight_buffer[(i, 0)] = base_weight * w;
            }
            let mut xtwx = weighted_xtx(cx, &weight_buffer);
            if lambda > 0.0 {
                add_ridge_to_diagonal(&mut xtwx, lambda, options.l2_penalty_exclude_intercept);
            }
            let xtw_rhs = weighted_xtz_with_buffer(cx, &weight_buffer, &z, &mut xtz_buffer);
            solve_with_stabilization(&xtwx, &xtw_rhs)?
        };

        let (beta_next, dev_next) = backtracking_update(
            cx,
            y,
            sample_weights,
            &beta,
            &beta_candidate,
            current_deviance,
            power,
            &mut search_eta,
            &mut search_mu,
        );

        let beta_converged = max_abs_diff(&beta_next, &beta) < options.tolerance;
        let dev_converged = relative_change(current_deviance, dev_next) < options.tolerance;
        if beta_converged || dev_converged {
            // Un-center back to the raw-x scale so SE/cov, predict and the marginal
            // standardisation all operate on the original design.
            let beta_raw = match &col_means {
                Some(means) => uncenter_beta(&beta_next, means),
                None => beta_next,
            };
            return finalize_fit(
                x,
                y,
                sample_weights,
                clusters,
                power,
                options,
                lambda,
                beta_raw,
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
        mean_y = weight.mul_add(y[(i, 0)], mean_y);
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
    // Saturation guard: a converged fit whose linear predictor still reaches the
    // `exp_clamped` ceiling has distorted, non-multiplicative predictions (the marginal
    // ratio collapses toward 1 and the counterfactual means blow up). Report it as
    // non-convergence instead of a degenerate "ok" fit — under the Relaxed strategy this
    // retries with stronger ridge (which shrinks η back into range); if it still saturates
    // the fit fails loudly rather than exporting a meaningless estimate. Centering keeps η
    // well inside the range for well-posed data, so this only fires on genuine divergence.
    if max_abs_linear_predictor(x, &beta_final) >= LINEAR_PREDICTOR_CLIP {
        return Err(TweedieError::NonConvergence);
    }
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
    fit_tweedie_input_warm(input, power, options, None)
}

/// Fit a Tweedie GLM from a `ModelInput`, warm-starting the IRLS from `initial_beta`.
///
/// A bootstrap replicate is a perturbation of the full sample, so seeding the IRLS with
/// the full-sample coefficients converges in far fewer iterations to the same optimum.
///
/// # Errors
///
/// Returns `TweedieError` if inputs are malformed or the solver fails.
pub fn fit_tweedie_input_warm(
    input: &ModelInput,
    power: f64,
    options: TweedieOptions,
    initial_beta: Option<&Mat<f64>>,
) -> Result<(TweedieModel, TweedieReport), TweedieError> {
    input.validate().map_err(TweedieError::from)?;
    fit_tweedie_weighted(
        &input.design_matrix,
        &input.outcome,
        input.sample_weights.as_ref(),
        input.cluster_ids.as_deref(),
        power,
        options,
        // input.validate() above already scanned finiteness + negativity.
        true,
        initial_beta,
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
                result = (weight * 2.0).mul_add(mui, result);
            } else {
                result = (weight * 2.0).mul_add(yi * (yi / mui).ln() - (yi - mui), result);
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
            result = (weight * 2.0).mul_add(((yi - mui) / mui) - (yi / mui).ln(), result);
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
            deviance = (sample_weight_at(sample_weights, i) * 2.0)
                .mul_add(term1 - term2 + term3, deviance);
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
    // Sandwich score per observation is the GLM estimating-equation contribution
    //   u_i = x_i * (dμ/dη)/V(μ) * (y_i - μ_i) * base_weight.
    // `weights` carries the FISHER weight W_i = base * (dμ/dη)²/V(μ) (= base*μ^(2-power)
    // for this log-link Tweedie family), so the score weight is W_i/(dμ/dη) = W_i/μ_i.
    // Using the raw response residual (y-μ)*W_i instead would inflate the meat by a
    // factor of μ (≈ mean cost, 1e4–1e5 DKK), blowing up the robust SE — so divide by μ
    // to use the WORKING residual. μ is exp-clamped strictly positive.
    if let Some(clusters) = clusters {
        let mut cluster_sums: HashMap<u64, Vec<f64>> = HashMap::new();
        for i in 0..x.nrows() {
            let resid = (y[(i, 0)] - mu[(i, 0)]) * weights[(i, 0)] / mu[(i, 0)];
            let entry = cluster_sums
                .entry(clusters[i])
                .or_insert_with(|| vec![0.0; p]);
            for j in 0..p {
                entry[j] = resid.mul_add(x[(i, j)], entry[j]);
            }
        }
        for sum in cluster_sums.values() {
            add_outer_product_scaled(&mut meat, sum, 1.0);
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
    Ok(cov_t.transpose().to_owned())
}

/// Returns the accepted step's `(beta, deviance)`. The deviance is the one already
/// computed for the returned beta inside the line search, so the caller reuses it
/// instead of recomputing `x*beta`, `exp`, and the deviance a second time per IRLS
/// iteration (the value is identical — same beta, same deviance function).
#[allow(clippy::too_many_arguments)]
fn backtracking_update(
    x: &Mat<f64>,
    y: &Mat<f64>,
    sample_weights: Option<&Mat<f64>>,
    beta_current: &Mat<f64>,
    beta_candidate: &Mat<f64>,
    current_deviance: f64,
    power: f64,
    scratch_eta: &mut Mat<f64>,
    scratch_mu: &mut Mat<f64>,
) -> (Mat<f64>, f64) {
    let mut step = 1.0_f64;
    let mut proposal = beta_candidate.clone();
    let mut best_beta = beta_candidate.clone();
    let mut best_deviance = f64::INFINITY;
    for _ in 0..BACKTRACKING_STEPS {
        if (step - 1.0).abs() < f64::EPSILON {
            proposal.clone_from(beta_candidate);
        } else {
            blend_betas_into(&mut proposal, beta_current, beta_candidate, step);
        }
        let dev = trial_deviance(x, y, sample_weights, &proposal, power, scratch_eta, scratch_mu);
        if dev.is_finite() && dev <= current_deviance {
            return (proposal, dev);
        }
        if dev.is_finite() && dev < best_deviance {
            best_deviance = dev;
            best_beta.clone_from(&proposal);
        }
        step *= 0.5;
    }
    // No trial reduced the deviance. Report best_beta's deviance — recomputed when the
    // search never recorded a finite one (all steps non-finite), so the returned value
    // matches what the caller would have computed for best_beta.
    let dev = if best_deviance.is_finite() {
        best_deviance
    } else {
        trial_deviance(x, y, sample_weights, &best_beta, power, scratch_eta, scratch_mu)
    };
    (best_beta, dev)
}

/// Deviance at `beta`, reusing the caller's `scratch_eta`/`scratch_mu` columns so the
/// line search allocates nothing per trial step.
fn trial_deviance(
    x: &Mat<f64>,
    y: &Mat<f64>,
    sample_weights: Option<&Mat<f64>>,
    beta: &Mat<f64>,
    power: f64,
    scratch_eta: &mut Mat<f64>,
    scratch_mu: &mut Mat<f64>,
) -> f64 {
    matvec_into(scratch_eta, x, beta);
    for i in 0..scratch_eta.nrows() {
        scratch_mu[(i, 0)] = exp_clamped(scratch_eta[(i, 0)]);
    }
    weighted_deviance(y, scratch_mu, power, sample_weights)
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

    // Mean of a single-column prediction matrix (the g-computation counterfactual mean).
    fn mean_of(pred: &Mat<f64>) -> f64 {
        (0..pred.nrows()).map(|i| pred[(i, 0)]).sum::<f64>() / pred.nrows() as f64
    }

    // Set the exposure column (index 1) to a constant `value` for the whole design — the
    // counterfactual design the runner feeds the marginal standardisation.
    fn set_exposure(x: &Mat<f64>, value: f64) -> Mat<f64> {
        Mat::from_fn(x.nrows(), x.ncols(), |i, j| if j == 1 { value } else { x[(i, j)] })
    }

    #[test]
    fn gamma_marginal_ratio_equals_exposure_coefficient() {
        // Regression guard for the counterfactual-mean explosion (run_010726 fully_adjusted:
        // exposed/unexposed means ~35M DKK, marginal_ratio 1.007 vs exp(β_exposed)=4.85).
        // For a log-link gamma the g-computation ratio is EXACTLY exp(β_exposed) — unless the
        // fit saturates the η-clamp, which collapses the ratio toward 1. Design: intercept,
        // binary exposure (col 1), and a large-offset covariate (col 2 ~1000) that makes the
        // UNCENTERED fit ill-conditioned. With centering the identity must hold tightly and
        // the means must stay on the data scale.
        let n = 400;
        let x = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => f64::from(u32::try_from(i % 2).unwrap()),
            _ => 1000.0 + usize_to_f64(i % 50),
        });
        // log μ = 9 + 0.02·(cov-1025) + 1.3·exposed, with deterministic within-cell scatter.
        let y = Mat::from_fn(n, 1, |i, _| {
            let cov = 1000.0 + usize_to_f64(i % 50);
            let exposed = f64::from(u32::try_from(i % 2).unwrap());
            let lin = 1.3f64.mul_add(exposed, 0.02f64.mul_add(cov - 1025.0, 9.0));
            let perturb = [0.7, 1.0, 1.4][i % 3];
            lin.exp() * perturb
        });

        let (model, _report) =
            fit_tweedie(&x, &y, None, 2.0, TweedieOptions::default()).expect("fit");

        let mean_exposed = mean_of(&model.predict(&set_exposure(&x, 1.0)).mean);
        let mean_unexposed = mean_of(&model.predict(&set_exposure(&x, 0.0)).mean);
        let ratio = mean_exposed / mean_unexposed;
        let expected = model.beta[(1, 0)].exp();

        assert!(
            (ratio - expected).abs() / expected < 1e-6,
            "marginal ratio {ratio} != exp(beta_exposed) {expected} (clamp saturation?)"
        );
        // ...and the counterfactual means are on the data scale, not blown up past the clamp.
        let max_y = (0..n).map(|i| y[(i, 0)]).fold(0.0_f64, f64::max);
        assert!(
            mean_exposed.is_finite() && mean_exposed < 100.0 * max_y,
            "exposed counterfactual mean exploded: {mean_exposed} (max y {max_y})"
        );
    }

    #[test]
    fn gamma_fit_conditions_large_offset_covariate() {
        // Centering must keep η in the data-supported range for a covariate whose raw scale
        // (~1000) would otherwise ill-condition the fit and let η drift toward the ±30 clamp.
        // The fitted mean tracks the data mean and no row's |η| approaches the ceiling.
        let n = 300;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 { 1.0 } else { 1000.0 + usize_to_f64(i % 40) }
        });
        let y = Mat::from_fn(n, 1, |i, _| {
            let cov = 1000.0 + usize_to_f64(i % 40);
            0.01f64.mul_add(cov - 1020.0, 8.5).exp() * [0.8, 1.0, 1.2][i % 3]
        });

        let (model, _report) =
            fit_tweedie(&x, &y, None, 2.0, TweedieOptions::default()).expect("fit");

        let max_abs_eta = max_abs_linear_predictor(&x, &model.beta);
        assert!(
            max_abs_eta < LINEAR_PREDICTOR_CLIP,
            "linear predictor reached the clamp: max|eta| = {max_abs_eta}"
        );
        let mean_pred = mean_of(&model.predict(&x).mean);
        let mean_y = (0..n).map(|i| y[(i, 0)]).sum::<f64>() / n as f64;
        assert!(
            (mean_pred - mean_y).abs() / mean_y < 0.05,
            "fitted mean {mean_pred} does not track data mean {mean_y}"
        );
    }

    #[test]
    fn gamma_fit_fails_loudly_when_outcome_forces_clamp() {
        // Saturation guard: an outcome so large that its log-mean exceeds the ±30 clamp
        // (exp(34) ≈ 5.8e14) cannot be represented below the ceiling. The fit must report
        // an error rather than silently returning a degenerate model whose predictions are
        // pinned at exp(30) — the failure mode that produced the meaningless fully_adjusted
        // headline. Strict strategy (the default) surfaces it directly with no ridge retry.
        let n = 50;
        let x = Mat::from_fn(n, 1, |_, _| 1.0); // intercept-only
        let y = Mat::from_fn(n, 1, |_, _| 34.0_f64.exp());
        let result = fit_tweedie(&x, &y, None, 2.0, TweedieOptions::default());
        assert!(
            result.is_err(),
            "a fit that can only converge at a saturated linear predictor must fail loudly"
        );
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
    fn robust_se_is_scale_invariant_for_log_link_gamma() {
        // Regression guard: the sandwich score must use the WORKING residual (y-μ)/μ,
        // not the raw response residual (y-μ). Under a log link, scaling y by a constant
        // shifts only the intercept and leaves every coefficient's robust SE unchanged.
        // The old bug used the raw residual, so SE(c*y) ≈ c*SE(y) — which on cost-scale
        // outcomes (μ ~ 1e4–1e5 DKK) inflated the gamma SEs by ~1e4×.
        let n = 60;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { usize_to_f64(i % 6) });
        // Strictly positive, associated with the covariate, with within-level scatter so
        // the residuals (and hence the robust SE) are non-degenerate.
        let make_y = |scale: f64| {
            Mat::from_fn(n, 1, |i, _| {
                let lin = 0.4f64.mul_add(usize_to_f64(i % 6), 1.0);
                let perturb = 0.4f64.mul_add(usize_to_f64(i % 3) - 1.0, 1.0); // 0.6, 1.0, 1.4
                lin.exp() * perturb * scale
            })
        };
        let clusters: Vec<u64> = (0..n).map(|i| (i / 4) as u64).collect();
        let slope_se = |scale: f64| {
            let options = TweedieOptions::builder().robust_se(true).build();
            let (_m, report) =
                fit_tweedie(&x, &make_y(scale), Some(&clusters), 2.0, options).expect("fit");
            report.se.expect("se")[(1, 0)]
        };
        let se_base = slope_se(1.0);
        let se_scaled = slope_se(1000.0);
        // Scale-invariant to <1% (the raw-residual bug gives ~1000x here).
        assert!(
            (se_scaled - se_base).abs() / se_base < 0.01,
            "robust SE not scale-invariant: base={se_base} scaled={se_scaled}"
        );
        // ...and a sane log-scale magnitude, not cost-scale.
        assert!(se_base < 1.0, "robust SE implausibly large: {se_base}");
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

    #[test]
    fn power_two_factor_once_recovers_group_log_means() {
        // Gamma (power == 2) log-link with intercept + one binary covariate: the MLE is
        // exactly the per-group log-means. This exercises the cached-factor
        // (constant-weight) IRLS path end to end and pins it to the correct optimum.
        let n = 200;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                f64::from(u32::from(i % 2 == 1))
            }
        });
        // Group 0 (even rows) mean 10, group 1 (odd rows) mean 30.
        let y = Mat::from_fn(n, 1, |i, _| if i % 2 == 0 { 10.0 } else { 30.0 });

        let (model, report) =
            fit_tweedie(&x, &y, None, 2.0, TweedieOptions::default()).expect("power-2 fit");

        // Intercept -> ln(10); slope -> ln(30) - ln(10) = ln(3).
        assert!(
            (model.beta[(0, 0)] - 10f64.ln()).abs() < 1e-4,
            "intercept = {}",
            model.beta[(0, 0)]
        );
        assert!(
            (model.beta[(1, 0)] - 3f64.ln()).abs() < 1e-4,
            "slope = {}",
            model.beta[(1, 0)]
        );
        assert!(report.meta.converged);
    }

    #[test]
    fn power_two_factor_once_respects_sample_weights() {
        // With constant weights the cached factor is X'diag(w)X; up-weighting one group
        // must still recover that group's log-mean (the weights cancel within a group),
        // confirming the cached factor carries the prior weights correctly.
        let n = 120;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                f64::from(u32::from(i % 2 == 1))
            }
        });
        let y = Mat::from_fn(n, 1, |i, _| if i % 2 == 0 { 5.0 } else { 50.0 });
        let w = Mat::from_fn(n, 1, |i, _| if i % 2 == 0 { 3.0 } else { 1.0 });
        let input = ModelInput::new(x, y).with_sample_weights(w);

        let (model, _) =
            fit_tweedie_input(&input, 2.0, TweedieOptions::default()).expect("weighted power-2 fit");
        assert!((model.beta[(0, 0)] - 5f64.ln()).abs() < 1e-4);
        assert!((model.beta[(1, 0)] - (50f64.ln() - 5f64.ln())).abs() < 1e-4);
    }

    #[test]
    fn warm_start_matches_cold_start_and_converges_faster() {
        let n = 60;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 20.0 });
        let y = Mat::from_fn(n, 1, |i, _| 1.0 + usize_to_f64(i));
        let input = ModelInput::new(x, y);
        let options = TweedieOptions::default();

        let (cold, cold_report) = fit_tweedie_input(&input, 2.0, options).expect("cold fit");
        let (warm, warm_report) =
            fit_tweedie_input_warm(&input, 2.0, options, Some(&cold.beta)).expect("warm fit");

        // Same optimum to the IRLS convergence tolerance. (IRLS stops on a small STEP,
        // so cold and warm both land within tolerance of the unique MLE but not at the
        // identical point — they agree to ~the step tolerance, far below any reported
        // precision, which is exactly the warm-start guarantee.)
        for i in 0..cold.beta.nrows() {
            assert!(
                (cold.beta[(i, 0)] - warm.beta[(i, 0)]).abs() < 1e-3,
                "coefficient {i} diverged: cold={} warm={}",
                cold.beta[(i, 0)],
                warm.beta[(i, 0)]
            );
        }
        // Starting at the converged solution converges essentially immediately, and
        // never in more iterations than the cold start.
        assert!(warm_report.meta.iterations <= cold_report.meta.iterations);
        assert!(warm_report.meta.iterations <= 2);
    }
}
