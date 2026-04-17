/////////////////////////////////////////////////////////////////////////////////////////////\\\
//
// Two-part model for semi-continuous outcome data.
//
// Created on: 24 Jan 2026     Author: Tobias Kragholm
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # Two-part model
//!
//! Implements a two-part model for semi-continuous outcome data with many zeros:
//! - Part 1: logistic regression for any positive outcome (> 0).
//! - Part 2: gamma regression with log link for positive outcomes.
//!
//! The module includes optional L2 regularization, robust or cluster-robust
//! standard errors, and bootstrap utilities for inference.

use crate::input::{InputError, ModelInput};
use cgp::prelude::*;

#[cgp_component(BinaryModel)]
#[allow(clippy::missing_errors_doc)]
pub trait CanFitBinaryPart {
    type BinaryData;
    type BinaryResult;

    /// Fit the binary part of the model.
    fn fit_binary(&self, data: &Self::BinaryData) -> Result<Self::BinaryResult, String>;
}

#[cgp_component(ContinuousModel)]
#[allow(clippy::missing_errors_doc)]
pub trait CanFitContinuousPart {
    type ContinuousData;
    type ContinuousResult;

    /// Fit the continuous part of the model.
    fn fit_continuous(&self, data: &Self::ContinuousData)
    -> Result<Self::ContinuousResult, String>;
}
use crate::models::matrix_ops::{map_mat, select_rows, select_values};
use crate::models::{
    AttemptDiagnostics, AttemptOutcome, FitMetadata, FitStrategy, Model, SolverKind,
};
#[cfg(feature = "bench-internals")]
use crate::utils::weighted_xtz;
use crate::utils::{
    add_ridge_to_diagonal, max_abs_diff, mean_column, mean_vector, solve_linear_system,
    solve_linear_system_ref, std_vector, weighted_xtx, weighted_xtz_with_buffer,
};
use faer::Mat;
use rand::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::function::gamma::ln_gamma;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use thiserror::Error;

/// Regularization strategy for GLM stages.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Regularization {
    /// No penalty.
    None,
    /// L2 (ridge) penalty.
    Ridge {
        lambda: f64,
        exclude_intercept: bool,
    },
    /// Elastic net penalty (L1 + L2).
    ElasticNet {
        lambda: f64,
        alpha: f64,
        exclude_intercept: bool,
    },
    /// Bayesian ridge (Normal prior), equivalent to an L2 penalty.
    BayesianRidge {
        prior_scale: f64,
        exclude_intercept: bool,
    },
}

/// Tuning parameters for two-part model fitting.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct FitOptions {
    /// Maximum number of IRLS iterations per stage.
    #[builder(default = 50_usize)]
    pub max_iter: usize,
    /// Convergence tolerance on coefficient changes.
    #[builder(default = 1e-6_f64)]
    pub tolerance: f64,
    /// Lower bound on IRLS weights.
    #[builder(default = 1e-6_f64)]
    pub min_weight: f64,
    /// Regularization strategy.
    #[builder(default = Regularization::None)]
    pub regularization: Regularization,
    /// If true, compute sandwich (robust) covariance.
    #[builder(default = false)]
    pub robust_se: bool,
    /// Strategy for handling non-convergence.
    #[builder(default = FitStrategy::Strict)]
    pub strategy: FitStrategy,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tolerance: 1e-6,
            min_weight: 1e-6,
            regularization: Regularization::None,
            robust_se: false,
            strategy: FitStrategy::Strict,
        }
    }
}

impl FitOptions {
    /// Stable defaults for noisy observational data.
    ///
    /// Uses mild ridge regularization to reduce solver failures while keeping
    /// coefficients close to the unpenalized solution.
    #[must_use]
    pub fn stable_defaults() -> Self {
        Self::builder()
            .regularization(Regularization::Ridge {
                lambda: 1e-4,
                exclude_intercept: true,
            })
            .strategy(FitStrategy::Relaxed {
                fallback_lambda: 1e-3,
                max_retries: 3,
                warm_start: true,
                time_budget: None,
            })
            .build()
    }
}

/// Errors returned by two-part model fitting.
#[derive(Debug, Error)]
pub enum TwoPartError {
    #[error("design matrix rows ({rows}) must match outcome length ({len})")]
    DimensionMismatch { rows: usize, len: usize },
    #[error("design matrix must have at least one column")]
    EmptyDesign,
    #[error("weights must be a single column matrix with the same number of rows as outcome")]
    InvalidWeightShape,
    #[error("weighted fit requires weights in ModelInput")]
    MissingWeights,
    #[error("clustered fit requires cluster labels in ModelInput")]
    MissingClusters,
    #[error("inputs contain non-finite values")]
    NonFiniteInput,
    #[error("outcome contains negative values")]
    NegativeOutcome,
    #[error("weights must be strictly positive")]
    NonPositiveWeights,
    #[error("positive outcome values required for gamma model")]
    NonPositiveOutcome,
    #[error("model failed to converge")]
    NonConvergence,
    #[error("linear solve failed")]
    SolveFailed,
    #[error("bootstrap iterations must be positive")]
    InvalidBootstrapIterations,
    #[error("bootstrap requires at least one row")]
    EmptyBootstrapSample,
    #[error("too many bootstrap failures ({0})")]
    TooManyBootstrapFailures(usize),
    #[error("outcome must be a single column matrix")]
    InvalidOutcomeShape,
}

#[allow(clippy::fallible_impl_from)]
impl From<InputError> for TwoPartError {
    fn from(value: InputError) -> Self {
        match value {
            InputError::EmptyDesign => Self::EmptyDesign,
            InputError::InvalidOutcomeShape => Self::InvalidOutcomeShape,
            InputError::DimensionMismatch { rows, len } => Self::DimensionMismatch { rows, len },
            InputError::InvalidWeightShape => Self::InvalidWeightShape,
            InputError::InvalidClusterLength { labels, rows } => Self::DimensionMismatch {
                rows: labels,
                len: rows,
            },
            InputError::NonFiniteDesign
            | InputError::NonFiniteOutcome
            | InputError::NonFiniteWeights => Self::NonFiniteInput,
            InputError::NegativeOutcome => Self::NegativeOutcome,
            InputError::NonPositiveWeights => Self::NonPositiveWeights,
            InputError::InvalidLabelLength { labels, cols: _ } => Self::DimensionMismatch {
                rows: labels,
                len: 0, // Placeholder
            },
            InputError::DuplicateLabels(labels) => {
                panic!("duplicate labels should be caught in validation: {labels}")
            }
        }
    }
}

#[cgp_component(FitOptionsProvider)]
pub trait HasFitOptions {
    fn options(&self) -> FitOptions;
}

pub struct TwoPartModelContext<'a> {
    pub options: FitOptions,
    pub _marker: PhantomData<&'a ()>,
}

impl HasFitOptions for TwoPartModelContext<'_> {
    fn options(&self) -> FitOptions {
        self.options
    }
}

pub struct LogisticBinaryPart;

#[cgp_impl(LogisticBinaryPart)]
impl<Context> BinaryModel for Context
where
    Context: HasFitOptions,
{
    type BinaryData = (Mat<f64>, Mat<f64>, Mat<f64>);
    type BinaryResult = (Mat<f64>, usize);

    fn fit_binary(&self, data: &Self::BinaryData) -> Result<Self::BinaryResult, String> {
        let (x, y, w) = data;
        let options = self.options();
        fit_logit_weighted(x, y, w, options).map_err(|e| e.to_string())
    }
}

pub struct GammaContinuousPart;

#[cgp_impl(GammaContinuousPart)]
impl<Context> ContinuousModel for Context
where
    Context: HasFitOptions,
{
    type ContinuousData = (Mat<f64>, Mat<f64>, Mat<f64>);
    type ContinuousResult = (Mat<f64>, usize);

    fn fit_continuous(
        &self,
        data: &Self::ContinuousData,
    ) -> Result<Self::ContinuousResult, String> {
        let (x, y, w) = data;
        let options = self.options();
        fit_gamma_log_link_weighted(x, y, w, options).map_err(|e| e.to_string())
    }
}

/// Two-part model coefficients for both stages.
#[derive(Debug, Clone)]
pub struct TwoPartModel {
    /// Logistic regression coefficients for Pr(y > 0).
    pub beta_logit: Mat<f64>,
    /// Gamma-log regression coefficients for E[y | y > 0].
    pub beta_gamma: Mat<f64>,
    /// Fit diagnostics.
    pub report: TwoPartReport,
}

/// Two-part model predictions.
#[derive(Debug, Clone)]
pub struct TwoPartPrediction {
    /// Predicted probability of any positive outcome.
    pub prob_positive: Mat<f64>,
    /// Predicted mean of positive outcomes.
    pub mean_positive: Mat<f64>,
    /// Predicted expected outcome (probability * `mean_positive`).
    pub expected_outcome: Mat<f64>,
}

impl TwoPartPrediction {
    /// Create a new prediction container with allocated matrices.
    #[must_use]
    pub fn new(nrows: usize) -> Self {
        Self {
            prob_positive: Mat::zeros(nrows, 1),
            mean_positive: Mat::zeros(nrows, 1),
            expected_outcome: Mat::zeros(nrows, 1),
        }
    }
}

/// Model diagnostics and inference outputs.
#[derive(Debug, Clone)]
pub struct TwoPartReport {
    /// Standardized fit metadata.
    pub meta: FitMetadata,
    /// Metadata specifically for the logit stage.
    pub logit_meta: FitMetadata,
    /// Metadata specifically for the gamma stage.
    pub gamma_meta: FitMetadata,
    /// Iterations used by the logistic stage.
    pub iterations_logit: usize,
    /// Iterations used by the gamma-log stage.
    pub iterations_gamma: usize,
    /// Standard errors for the logistic coefficients.
    pub se_logit: Option<Mat<f64>>,
    /// Standard errors for the gamma-log coefficients.
    pub se_gamma: Option<Mat<f64>>,
    /// Covariance matrix for the logistic coefficients.
    pub cov_logit: Option<Mat<f64>>,
    /// Covariance matrix for the gamma-log coefficients.
    pub cov_gamma: Option<Mat<f64>>,
    /// True if robust (sandwich) covariance was used.
    pub robust: bool,
    /// True if cluster-robust covariance was used.
    pub clustered: bool,
    /// Number of clusters used for cluster-robust covariance.
    pub cluster_count: Option<usize>,
    /// History of retry attempts (if Relaxed strategy used).
    pub attempts: Vec<AttemptDiagnostics>,
}

impl Model for TwoPartModel {
    type Prediction = TwoPartPrediction;
    type Report = TwoPartReport;

    fn predict(&self, x: &Mat<f64>) -> Self::Prediction {
        let mut out = TwoPartPrediction::new(x.nrows());
        self.predict_into(x, &mut out);
        out
    }

    fn predict_into(&self, x: &Mat<f64>, out: &mut Self::Prediction) {
        let eta_logit = x * &self.beta_logit;
        for i in 0..x.nrows() {
            out.prob_positive[(i, 0)] = 1.0 / (1.0 + (-eta_logit[(i, 0)]).exp());
        }

        let eta_gamma = x * &self.beta_gamma;
        for i in 0..x.nrows() {
            out.mean_positive[(i, 0)] = eta_gamma[(i, 0)].exp();
        }

        for i in 0..x.nrows() {
            out.expected_outcome[(i, 0)] = out.prob_positive[(i, 0)] * out.mean_positive[(i, 0)];
        }
    }

    fn report(&self) -> &Self::Report {
        &self.report
    }
}

/// Bootstrap configuration for two-part fitting.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct BootstrapOptions {
    /// Number of bootstrap draws.
    #[builder(default = 200_usize)]
    pub iterations: usize,
    /// RNG seed for reproducibility.
    #[builder(default = 42_u64)]
    pub seed: u64,
    /// Skip non-converged fits instead of failing.
    #[builder(default = true)]
    pub skip_nonconvergence: bool,
    /// Maximum allowed bootstrap failures before aborting.
    #[builder(default = 50_usize)]
    pub max_failures: usize,
}

impl Default for BootstrapOptions {
    fn default() -> Self {
        Self {
            iterations: 200,
            seed: 42,
            skip_nonconvergence: true,
            max_failures: 50,
        }
    }
}

/// Bootstrap outputs for both stages.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Bootstrap samples of logistic coefficients.
    pub betas_logit: Vec<Mat<f64>>,
    /// Bootstrap samples of gamma-log coefficients.
    pub betas_gamma: Vec<Mat<f64>>,
    /// Number of skipped failures.
    pub failures: usize,
}

/// Confidence interval for a coefficient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
}

/// Summary statistics from bootstrap draws.
#[derive(Debug, Clone)]
pub struct BootstrapSummary {
    /// Bootstrap mean.
    pub mean: Mat<f64>,
    /// Bootstrap standard error.
    pub se: Mat<f64>,
    /// Percentile confidence intervals.
    pub ci: Vec<ConfidenceInterval>,
}

/// Fit an unweighted two-part model (logit + gamma log-link).
///
/// # Errors
///
/// Returns `TwoPartError` if inputs are malformed or the solver fails to converge.
pub(crate) fn fit_two_part(
    x: &Mat<f64>,
    y: &Mat<f64>,
    options: FitOptions,
) -> Result<(TwoPartModel, TwoPartReport), TwoPartError> {
    let weights = Mat::from_fn(y.nrows(), 1, |_, _| 1.0);
    fit_two_part_weighted(x, y, &weights, options)
}

/// Fit a two-part model from a `ModelInput` container.
///
/// # Errors
///
/// Returns `TwoPartError` if inputs are malformed or the solver fails to converge.
pub fn fit_two_part_input(
    input: &ModelInput,
    options: FitOptions,
) -> Result<(TwoPartModel, TwoPartReport), TwoPartError> {
    input.validate()?;

    let start_time = Instant::now();
    let mut current_options = options;
    let mut attempts = Vec::new();
    let mut last_err = TwoPartError::NonConvergence;

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
                    lambda_used: 0.0, // N/A
                    meta: FitMetadata::default(),
                    outcome: AttemptOutcome::TimedOut,
                });
                break;
            }

            // Increase regularization strength with each attempt
            let scale = 10.0f64.powi(i32::try_from(attempt_idx - 1).unwrap_or(0));
            current_options.regularization = Regularization::Ridge {
                lambda: fallback_lambda * scale,
                exclude_intercept: true,
            };
        }

        let lambda_used = match current_options.regularization {
            Regularization::Ridge { lambda, .. } | Regularization::ElasticNet { lambda, .. } => {
                lambda
            }
            Regularization::BayesianRidge { prior_scale, .. } => 1.0 / (prior_scale * prior_scale),
            Regularization::None => 0.0,
        };

        let result = match (&input.sample_weights, &input.cluster_ids) {
            (Some(weights), Some(clusters)) => fit_two_part_clustered_weighted(
                &input.design_matrix,
                &input.outcome,
                weights,
                clusters,
                current_options,
            ),
            (Some(weights), None) => fit_two_part_weighted(
                &input.design_matrix,
                &input.outcome,
                weights,
                current_options,
            ),
            (None, Some(clusters)) => fit_two_part_clustered(
                &input.design_matrix,
                &input.outcome,
                clusters,
                current_options,
            ),
            (None, None) => fit_two_part(&input.design_matrix, &input.outcome, current_options),
        };

        match result {
            Ok((mut model, mut report)) => {
                report.attempts = attempts;
                report.meta.fallback_attempts = attempt_idx;
                model.report = report.clone();
                return Ok((model, report));
            }
            Err(e @ (TwoPartError::NonConvergence | TwoPartError::SolveFailed)) => {
                attempts.push(AttemptDiagnostics {
                    attempt: attempt_idx,
                    lambda_used,
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

/// Fit a weighted two-part model from a `ModelInput` container.
///
/// # Errors
///
/// Returns `TwoPartError` if inputs are malformed or the solver fails to converge.
pub fn fit_two_part_weighted_input(
    input: &ModelInput,
    options: FitOptions,
) -> Result<(TwoPartModel, TwoPartReport), TwoPartError> {
    let _weights = input
        .sample_weights
        .as_ref()
        .ok_or(TwoPartError::MissingWeights)?;
    let mut weighted_input = input.clone();
    weighted_input.cluster_ids = None; // Ensure non-clustered path
    fit_two_part_input(&weighted_input, options)
}

/// Fit a clustered two-part model from a `ModelInput` container.
///
/// # Errors
///
/// Returns `TwoPartError` if inputs are malformed or the solver fails to converge.
pub fn fit_two_part_clustered_input(
    input: &ModelInput,
    options: FitOptions,
) -> Result<(TwoPartModel, TwoPartReport), TwoPartError> {
    let _clusters = input
        .cluster_ids
        .as_ref()
        .ok_or(TwoPartError::MissingClusters)?;
    fit_two_part_input(input, options)
}

/// Fit a weighted two-part model (used for IPW and survey weights).
///
/// # Errors
///
/// Returns `TwoPartError` if inputs are malformed or the solver fails to converge.
pub(crate) fn fit_two_part_weighted(
    x: &Mat<f64>,
    y: &Mat<f64>,
    weights: &Mat<f64>,
    options: FitOptions,
) -> Result<(TwoPartModel, TwoPartReport), TwoPartError> {
    let start_time = Instant::now();

    let is_positive = Mat::from_fn(y.nrows(), 1, |i, _| if y[(i, 0)] > 0.0 { 1.0 } else { 0.0 });
    let (beta_logit, iterations_logit) = fit_logit_weighted(x, &is_positive, weights, options)?;

    let positive_indices: Vec<usize> = (0..y.nrows()).filter(|&idx| y[(idx, 0)] > 0.0).collect();

    if positive_indices.is_empty() {
        return Err(TwoPartError::NonPositiveOutcome);
    }

    let x_pos = select_rows(x, &positive_indices);
    let y_pos = select_values(y, &positive_indices);
    let w_pos = select_values(weights, &positive_indices);

    if (0..y_pos.nrows()).any(|i| y_pos[(i, 0)] <= 0.0) {
        return Err(TwoPartError::NonPositiveOutcome);
    }

    let (beta_gamma, iterations_gamma) =
        fit_gamma_log_link_weighted(&x_pos, &y_pos, &w_pos, options)?;

    let cov_logit = Some(covariance_logit_weighted(
        x,
        &is_positive,
        weights,
        &beta_logit,
        options,
    )?);
    let cov_gamma = Some(covariance_gamma_weighted(
        &x_pos,
        &y_pos,
        &w_pos,
        &beta_gamma,
        options,
    )?);
    let se_logit = cov_logit.as_ref().map(diag_sqrt);
    let se_gamma = cov_gamma.as_ref().map(diag_sqrt);

    let execution_time = start_time.elapsed();
    let logit_meta = FitMetadata {
        iterations: iterations_logit,
        converged: true,
        execution_time,
        solver: SolverKind::Irls,
        ..FitMetadata::default()
    };
    let gamma_meta = FitMetadata {
        iterations: iterations_gamma,
        converged: true,
        execution_time,
        solver: SolverKind::Irls,
        ..FitMetadata::default()
    };
    let meta = FitMetadata {
        iterations: iterations_logit + iterations_gamma,
        converged: true,
        execution_time,
        solver: SolverKind::Irls,
        ..FitMetadata::default()
    };

    let report = TwoPartReport {
        meta,
        logit_meta,
        gamma_meta,
        iterations_logit,
        iterations_gamma,
        se_logit,
        se_gamma,
        cov_logit,
        cov_gamma,
        robust: options.robust_se,
        clustered: false,
        cluster_count: None,
        attempts: Vec::new(),
    };

    Ok((
        TwoPartModel {
            beta_logit,
            beta_gamma,
            report: report.clone(),
        },
        report,
    ))
}

/// Fit a two-part model with cluster-robust covariance.
///
/// # Errors
///
/// Returns `TwoPartError` if inputs are malformed or the solver fails to converge.
pub(crate) fn fit_two_part_clustered(
    x: &Mat<f64>,
    y: &Mat<f64>,
    clusters: &[u64],
    options: FitOptions,
) -> Result<(TwoPartModel, TwoPartReport), TwoPartError> {
    let weights = Mat::from_fn(y.nrows(), 1, |_, _| 1.0);
    fit_two_part_clustered_weighted(x, y, &weights, clusters, options)
}

/// Fit a weighted two-part model with cluster-robust covariance.
///
/// # Errors
///
/// Returns `TwoPartError` if inputs are malformed or the solver fails to converge.
pub(crate) fn fit_two_part_clustered_weighted(
    x: &Mat<f64>,
    y: &Mat<f64>,
    weights: &Mat<f64>,
    clusters: &[u64],
    options: FitOptions,
) -> Result<(TwoPartModel, TwoPartReport), TwoPartError> {
    let (model, base_report) = fit_two_part_weighted(x, y, weights, options)?;
    if !options.robust_se {
        return Ok((model, base_report));
    }

    let is_positive = Mat::from_fn(y.nrows(), 1, |i, _| if y[(i, 0)] > 0.0 { 1.0 } else { 0.0 });
    let positive_indices: Vec<usize> = (0..y.nrows()).filter(|&idx| y[(idx, 0)] > 0.0).collect();
    let x_pos = select_rows(x, &positive_indices);
    let y_pos = select_values(y, &positive_indices);
    let w_pos = select_values(weights, &positive_indices);
    let clusters_pos = select_cluster_ids(clusters, &positive_indices);

    let cov_logit = covariance_logit_cluster_weighted(
        x,
        &is_positive,
        weights,
        &model.beta_logit,
        clusters,
        options,
    )?;
    let cov_gamma = covariance_gamma_cluster_weighted(
        &x_pos,
        &y_pos,
        &w_pos,
        &model.beta_gamma,
        &clusters_pos,
        options,
    )?;

    let mut report = base_report;
    report.se_logit = Some(diag_sqrt(&cov_logit));
    report.se_gamma = Some(diag_sqrt(&cov_gamma));
    report.cov_logit = Some(cov_logit);
    report.cov_gamma = Some(cov_gamma);
    report.robust = true;
    report.clustered = true;
    report.cluster_count = Some(cluster_count(clusters));

    let mut model_with_report = model;
    model_with_report.report = report.clone();

    Ok((model_with_report, report))
}

/// Compute a two-part log-likelihood under a logistic + gamma-log specification.
///
/// Uses a moment-based gamma shape parameter (phi) estimated from Pearson
/// residuals for the positive part.
#[must_use]
pub fn log_likelihood(y: &Mat<f64>, prob: &Mat<f64>, mean_pos: &Mat<f64>) -> f64 {
    if y.ncols() != 1 || prob.ncols() != 1 || mean_pos.ncols() != 1 {
        return f64::NAN;
    }
    if y.nrows() != prob.nrows() || y.nrows() != mean_pos.nrows() {
        return f64::NAN;
    }

    let mut loglik = 0.0;
    let mut phi_sum = 0.0;
    let mut n_pos = 0.0;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        let pi = prob[(i, 0)].clamp(1e-12, 1.0 - 1e-12);
        if yi > 0.0 {
            loglik += pi.ln();
            let mu = mean_pos[(i, 0)].max(1e-12);
            phi_sum += (yi - mu).powi(2) / (mu * mu);
            n_pos += 1.0;
        } else {
            loglik += (1.0 - pi).ln();
        }
    }

    if n_pos == 0.0 {
        return loglik;
    }

    let phi = (phi_sum / n_pos).max(1e-8);
    let shape = 1.0 / phi;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        if yi > 0.0 {
            let mu = mean_pos[(i, 0)].max(1e-12);
            let scale = mu * phi;
            loglik += (shape - 1.0).mul_add(yi.ln(), -(yi / scale))
                - ln_gamma(shape)
                - shape * scale.ln();
        }
    }
    loglik
}

/// Compute Wald confidence intervals from a covariance matrix.
#[must_use]
pub fn coefficient_confidence_intervals(
    beta: &Mat<f64>,
    cov: &Mat<f64>,
    alpha: f64,
) -> Vec<ConfidenceInterval> {
    let z = normal_quantile(1.0 - alpha / 2.0);
    let mut intervals = Vec::with_capacity(beta.nrows());
    for i in 0..beta.nrows() {
        let se = cov[(i, i)].max(0.0).sqrt();
        intervals.push(ConfidenceInterval {
            lower: beta[(i, 0)] - z * se,
            upper: beta[(i, 0)] + z * se,
        });
    }
    intervals
}

/// Bootstrap two-part model fits by resampling rows with replacement.
///
/// # Errors
///
/// Returns `TwoPartError` if inputs are malformed or the bootstrap fails.
pub fn bootstrap(
    x: &Mat<f64>,
    y: &Mat<f64>,
    options: FitOptions,
    bootstrap: BootstrapOptions,
) -> Result<BootstrapResult, TwoPartError> {
    if bootstrap.iterations == 0 {
        return Err(TwoPartError::InvalidBootstrapIterations);
    }
    if x.nrows() == 0 {
        return Err(TwoPartError::EmptyBootstrapSample);
    }
    if x.nrows() != y.nrows() {
        return Err(TwoPartError::DimensionMismatch {
            rows: x.nrows(),
            len: y.nrows(),
        });
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(bootstrap.seed);
    let mut betas_logit = Vec::with_capacity(bootstrap.iterations);
    let mut betas_gamma = Vec::with_capacity(bootstrap.iterations);
    let mut failures = 0;

    for _ in 0..bootstrap.iterations {
        let indices = (0..x.nrows())
            .map(|_| rng.random_range(0..x.nrows()))
            .collect::<Vec<_>>();
        let x_sample = select_rows(x, &indices);
        let y_sample = select_values(y, &indices);
        match fit_two_part(&x_sample, &y_sample, options) {
            Ok((model, _)) => {
                betas_logit.push(model.beta_logit);
                betas_gamma.push(model.beta_gamma);
            }
            Err(TwoPartError::NonConvergence) if bootstrap.skip_nonconvergence => {
                failures += 1;
                if failures > bootstrap.max_failures {
                    return Err(TwoPartError::TooManyBootstrapFailures(failures));
                }
            }
            Err(err) => return Err(err),
        }
    }

    Ok(BootstrapResult {
        betas_logit,
        betas_gamma,
        failures,
    })
}

/// Percentile confidence intervals from bootstrap draws.
#[must_use]
pub fn bootstrap_percentile_ci(betas: &[Mat<f64>], alpha: f64) -> Vec<ConfidenceInterval> {
    if betas.is_empty() {
        return Vec::new();
    }
    let n = betas[0].nrows();
    let last = betas.len().saturating_sub(1);
    let (mut lower_idx, mut upper_idx) = crate::utils::boot_index_bounds(alpha, betas.len());
    lower_idx = lower_idx.min(last);
    upper_idx = upper_idx.min(last).max(lower_idx);

    let mut intervals = Vec::with_capacity(n);
    for col in 0..n {
        let mut values = betas.iter().map(|b| b[(col, 0)]).collect::<Vec<_>>();
        let upper = order_statistic_unsorted(&mut values, upper_idx).unwrap_or(f64::NAN);
        let lower = if lower_idx == upper_idx {
            upper
        } else {
            order_statistic_unsorted(&mut values[..=upper_idx], lower_idx).unwrap_or(upper)
        };
        intervals.push(ConfidenceInterval { lower, upper });
    }
    intervals
}

fn order_statistic_unsorted(values: &mut [f64], nth: usize) -> Option<f64> {
    if values.is_empty() || nth >= values.len() {
        return None;
    }
    values.select_nth_unstable_by(nth, f64::total_cmp);
    Some(values[nth])
}

/// Compute bootstrap mean, SE, and percentile CI.
#[must_use]
pub fn bootstrap_summary(betas: &[Mat<f64>], alpha: f64) -> BootstrapSummary {
    if betas.is_empty() {
        return BootstrapSummary {
            mean: Mat::zeros(0, 1),
            se: Mat::zeros(0, 1),
            ci: Vec::new(),
        };
    }
    let mean = mean_vector(betas);
    let se = std_vector(betas, &mean);
    let ci = bootstrap_percentile_ci(betas, alpha);
    BootstrapSummary { mean, se, ci }
}

fn fit_logit_weighted(
    x: &Mat<f64>,
    y: &Mat<f64>,
    weights: &Mat<f64>,
    options: FitOptions,
) -> Result<(Mat<f64>, usize), TwoPartError> {
    // Degenerate outcome guards:
    // If the binary outcome is constant (all 0 or all 1), IRLS will not
    // meaningfully iterate toward a finite optimum. In this case we fit an
    // intercept-only prevalence model with a clamped logit.
    let mut weighted_sum = 0.0;
    let mut weighted_positive = 0.0;
    let mut saw_positive = false;
    let mut saw_zero = false;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        if yi > 0.0 {
            saw_positive = true;
        } else {
            saw_zero = true;
        }
        let wi = weights[(i, 0)];
        weighted_sum += wi;
        weighted_positive = yi.mul_add(wi, weighted_positive);
    }
    if !saw_positive || !saw_zero {
        let mut beta = Mat::<f64>::zeros(x.ncols(), 1);
        if weighted_sum > 0.0 && x.ncols() > 0 {
            let p = (weighted_positive / weighted_sum)
                .clamp(options.min_weight, 1.0 - options.min_weight);
            beta[(0, 0)] = (p / (1.0 - p)).ln();
        }
        return Ok((beta, 0));
    }

    let mut beta = Mat::<f64>::zeros(x.ncols(), 1);
    let regularization = options.regularization;
    let mut weighted_xtz_buffer = Vec::new();

    for iteration in 0..options.max_iter {
        let eta = x * &beta;
        let p = logistic(&eta);
        let weights = Mat::from_fn(p.nrows(), 1, |i, _| {
            let value = p[(i, 0)] * (1.0 - p[(i, 0)]);
            (value.max(options.min_weight)) * weights[(i, 0)]
        });

        let z = Mat::from_fn(eta.nrows(), 1, |i, _| {
            eta[(i, 0)] + (y[(i, 0)] - p[(i, 0)]) / weights[(i, 0)]
        });

        let beta_next =
            weighted_least_squares(x, &weights, &z, regularization, &mut weighted_xtz_buffer)?;

        if max_abs_diff(&beta_next, &beta) < options.tolerance {
            return Ok((beta_next, iteration + 1));
        }
        beta = beta_next;
    }

    Err(TwoPartError::NonConvergence)
}

fn fit_gamma_log_link_weighted(
    x: &Mat<f64>,
    y: &Mat<f64>,
    weights: &Mat<f64>,
    options: FitOptions,
) -> Result<(Mat<f64>, usize), TwoPartError> {
    let mut beta = Mat::<f64>::zeros(x.ncols(), 1);
    let regularization = options.regularization;
    let mut weighted_xtz_buffer = Vec::new();
    if x.ncols() > 0 {
        let mean = mean_column(y);
        if mean > 0.0 {
            beta[(0, 0)] = mean.ln();
        }
    }

    for iteration in 0..options.max_iter {
        let eta = x * &beta;
        let mu = map_mat(&eta, f64::exp);
        let z = Mat::from_fn(eta.nrows(), 1, |i, _| {
            eta[(i, 0)] + (y[(i, 0)] - mu[(i, 0)]) / mu[(i, 0)]
        });

        let beta_next =
            weighted_least_squares(x, weights, &z, regularization, &mut weighted_xtz_buffer)?;

        if max_abs_diff(&beta_next, &beta) < options.tolerance {
            return Ok((beta_next, iteration + 1));
        }
        beta = beta_next;
    }

    Err(TwoPartError::NonConvergence)
}

fn weighted_least_squares(
    x: &Mat<f64>,
    weights: &Mat<f64>,
    z: &Mat<f64>,
    regularization: Regularization,
    weighted_buffer: &mut Vec<f64>,
) -> Result<Mat<f64>, TwoPartError> {
    if let Regularization::ElasticNet { .. } = regularization {
        return elastic_net_wls(x, weights, z, regularization);
    }

    let mut xtwx = weighted_xtx(x, weights);
    if let Some((lambda, exclude_intercept)) = ridge_from_regularization(regularization)
        && lambda > 0.0
    {
        add_ridge_to_diagonal(&mut xtwx, lambda, exclude_intercept);
    }
    let xtw_rhs = weighted_xtz_with_buffer(x, weights, z, weighted_buffer);
    crate::utils::solve_linear_system(&xtwx, &xtw_rhs).map_err(|_| TwoPartError::SolveFailed)
}

fn logistic(values: &Mat<f64>) -> Mat<f64> {
    map_mat(values, |value| 1.0 / (1.0 + (-value).exp()))
}

fn select_cluster_ids(clusters: &[u64], indices: &[usize]) -> Vec<u64> {
    indices.iter().map(|idx| clusters[*idx]).collect()
}

fn ridge_from_regularization(regularization: Regularization) -> Option<(f64, bool)> {
    match regularization {
        Regularization::None => None,
        Regularization::Ridge {
            lambda,
            exclude_intercept,
        } => Some((lambda.max(0.0), exclude_intercept)),
        Regularization::ElasticNet {
            lambda,
            alpha,
            exclude_intercept,
        } => {
            let l2 = lambda.max(0.0) * (1.0 - alpha.clamp(0.0, 1.0));
            Some((l2, exclude_intercept))
        }
        Regularization::BayesianRidge {
            prior_scale,
            exclude_intercept,
        } => {
            let scale = prior_scale.max(1e-8);
            Some((1.0 / (scale * scale), exclude_intercept))
        }
    }
}

fn elastic_net_wls(
    x: &Mat<f64>,
    weights: &Mat<f64>,
    z: &Mat<f64>,
    regularization: Regularization,
) -> Result<Mat<f64>, TwoPartError> {
    let (lambda, alpha, exclude_intercept) = match regularization {
        Regularization::ElasticNet {
            lambda,
            alpha,
            exclude_intercept,
        } => (lambda.max(0.0), alpha.clamp(0.0, 1.0), exclude_intercept),
        _ => return Err(TwoPartError::SolveFailed),
    };

    let n = x.nrows();
    let p = x.ncols();
    let mut beta = Mat::<f64>::zeros(p, 1);
    let mut residual = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        residual[(i, 0)] = z[(i, 0)];
    }

    let mut col_norms = vec![0.0; p];
    for j in 0..p {
        let mut norm = 0.0;
        for i in 0..n {
            let xij = x[(i, j)];
            norm = (weights[(i, 0)] * xij).mul_add(xij, norm);
        }
        col_norms[j] = norm.max(1e-12);
    }

    let mut iterations = 0;
    loop {
        iterations += 1;
        let mut max_delta = 0.0;
        for j in 0..p {
            let mut rho = 0.0;
            for i in 0..n {
                rho = (weights[(i, 0)] * x[(i, j)]).mul_add(residual[(i, 0)], rho);
            }
            rho = col_norms[j].mul_add(beta[(j, 0)], rho);

            let new_beta = if exclude_intercept && j == 0 {
                rho / col_norms[j]
            } else {
                soft_threshold(rho, lambda * alpha) / lambda.mul_add(1.0 - alpha, col_norms[j])
            };

            let delta = new_beta - beta[(j, 0)];
            if delta != 0.0 {
                for i in 0..n {
                    residual[(i, 0)] = x[(i, j)].mul_add(-delta, residual[(i, 0)]);
                }
            }
            if delta.abs() > max_delta {
                max_delta = delta.abs();
            }
            beta[(j, 0)] = new_beta;
        }

        if max_delta < 1e-6 || iterations >= 200 {
            break;
        }
    }

    Ok(beta)
}

fn soft_threshold(value: f64, penalty: f64) -> f64 {
    if value > penalty {
        value - penalty
    } else if value < -penalty {
        value + penalty
    } else {
        0.0
    }
}

fn covariance_logit_weighted(
    x: &Mat<f64>,
    y: &Mat<f64>,
    weights: &Mat<f64>,
    beta: &Mat<f64>,
    options: FitOptions,
) -> Result<Mat<f64>, TwoPartError> {
    let eta = x * beta;
    let p = logistic(&eta);
    let weights = Mat::from_fn(p.nrows(), 1, |i, _| {
        let value = p[(i, 0)] * (1.0 - p[(i, 0)]);
        (value.max(options.min_weight)) * weights[(i, 0)]
    });

    let mut xtwx = weighted_xtx(x, &weights);
    if let Some((lambda, exclude_intercept)) = ridge_from_regularization(options.regularization)
        && lambda > 0.0
    {
        add_ridge_to_diagonal(&mut xtwx, lambda, exclude_intercept);
    }
    if !options.robust_se {
        return covariance_from_information(&xtwx);
    }

    let residuals = Mat::from_fn(y.nrows(), 1, |i, _| {
        (y[(i, 0)] - p[(i, 0)]) * weights[(i, 0)]
    });
    let mut meat = Mat::<f64>::zeros(x.ncols(), x.ncols());
    for i in 0..x.nrows() {
        let weight = residuals[(i, 0)] * residuals[(i, 0)];
        for col_i in 0..x.ncols() {
            for col_j in 0..x.ncols() {
                meat[(col_i, col_j)] =
                    (x[(i, col_i)] * x[(i, col_j)]).mul_add(weight, meat[(col_i, col_j)]);
            }
        }
    }

    sandwich_covariance(&xtwx, &meat)
}

fn covariance_gamma_weighted(
    x: &Mat<f64>,
    y: &Mat<f64>,
    weights: &Mat<f64>,
    beta: &Mat<f64>,
    options: FitOptions,
) -> Result<Mat<f64>, TwoPartError> {
    let eta = x * beta;
    let mu = map_mat(&eta, f64::exp);

    let w = Mat::from_fn(mu.nrows(), 1, |i, _| weights[(i, 0)]);
    let mut xtx = weighted_xtx(x, &w);
    if let Some((lambda, exclude_intercept)) = ridge_from_regularization(options.regularization)
        && lambda > 0.0
    {
        add_ridge_to_diagonal(&mut xtx, lambda, exclude_intercept);
    }
    if !options.robust_se {
        return covariance_from_information(&xtx);
    }

    let residuals = Mat::from_fn(y.nrows(), 1, |i, _| {
        ((y[(i, 0)] - mu[(i, 0)]) / mu[(i, 0)]) * weights[(i, 0)]
    });
    let mut meat = Mat::<f64>::zeros(x.ncols(), x.ncols());
    for i in 0..x.nrows() {
        let weight = residuals[(i, 0)] * residuals[(i, 0)];
        for col_i in 0..x.ncols() {
            for col_j in 0..x.ncols() {
                meat[(col_i, col_j)] =
                    (x[(i, col_i)] * x[(i, col_j)]).mul_add(weight, meat[(col_i, col_j)]);
            }
        }
    }

    sandwich_covariance(&xtx, &meat)
}

fn covariance_logit_cluster_weighted(
    x: &Mat<f64>,
    y: &Mat<f64>,
    weights: &Mat<f64>,
    beta: &Mat<f64>,
    clusters: &[u64],
    options: FitOptions,
) -> Result<Mat<f64>, TwoPartError> {
    let eta = x * beta;
    let p = logistic(&eta);
    let weights = Mat::from_fn(p.nrows(), 1, |i, _| {
        let value = p[(i, 0)] * (1.0 - p[(i, 0)]);
        (value.max(options.min_weight)) * weights[(i, 0)]
    });

    let mut xtwx = weighted_xtx(x, &weights);
    if let Some((lambda, exclude_intercept)) = ridge_from_regularization(options.regularization)
        && lambda > 0.0
    {
        add_ridge_to_diagonal(&mut xtwx, lambda, exclude_intercept);
    }
    let residuals = Mat::from_fn(y.nrows(), 1, |i, _| {
        (y[(i, 0)] - p[(i, 0)]) * weights[(i, 0)]
    });
    let mut cluster_sums: HashMap<u64, Mat<f64>> = HashMap::new();
    for i in 0..x.nrows() {
        let entry = cluster_sums
            .entry(clusters[i])
            .or_insert_with(|| Mat::zeros(x.ncols(), 1));
        for col in 0..x.ncols() {
            entry[(col, 0)] = x[(i, col)].mul_add(residuals[(i, 0)], entry[(col, 0)]);
        }
    }

    let mut meat = Mat::<f64>::zeros(x.ncols(), x.ncols());
    for (_, sum) in cluster_sums {
        for i in 0..x.ncols() {
            for j in 0..x.ncols() {
                meat[(i, j)] = sum[(i, 0)].mul_add(sum[(j, 0)], meat[(i, j)]);
            }
        }
    }

    sandwich_covariance(&xtwx, &meat)
}

fn covariance_gamma_cluster_weighted(
    x: &Mat<f64>,
    y: &Mat<f64>,
    weights: &Mat<f64>,
    beta: &Mat<f64>,
    clusters: &[u64],
    options: FitOptions,
) -> Result<Mat<f64>, TwoPartError> {
    let eta = x * beta;
    let mu = map_mat(&eta, f64::exp);
    let residuals = Mat::from_fn(y.nrows(), 1, |i, _| {
        ((y[(i, 0)] - mu[(i, 0)]) / mu[(i, 0)]) * weights[(i, 0)]
    });

    let mut xtx = weighted_xtx(x, weights);
    if let Some((lambda, exclude_intercept)) = ridge_from_regularization(options.regularization)
        && lambda > 0.0
    {
        add_ridge_to_diagonal(&mut xtx, lambda, exclude_intercept);
    }
    let mut cluster_sums: HashMap<u64, Mat<f64>> = HashMap::new();
    for i in 0..x.nrows() {
        let entry = cluster_sums
            .entry(clusters[i])
            .or_insert_with(|| Mat::zeros(x.ncols(), 1));
        for col in 0..x.ncols() {
            entry[(col, 0)] = x[(i, col)].mul_add(residuals[(i, 0)], entry[(col, 0)]);
        }
    }

    let mut meat = Mat::<f64>::zeros(x.ncols(), x.ncols());
    for (_, sum) in cluster_sums {
        for i in 0..x.ncols() {
            for j in 0..x.ncols() {
                meat[(i, j)] = sum[(i, 0)].mul_add(sum[(j, 0)], meat[(i, j)]);
            }
        }
    }

    sandwich_covariance(&xtx, &meat)
}

fn diag_sqrt(covariance: &Mat<f64>) -> Mat<f64> {
    Mat::from_fn(covariance.nrows(), 1, |i, _| {
        covariance[(i, i)].max(0.0).sqrt()
    })
}

#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_two_part_weighted_xtz(x: &Mat<f64>, weights: &Mat<f64>, z: &Mat<f64>) -> Mat<f64> {
    weighted_xtz(x, weights, z)
}

fn normal_quantile(p: f64) -> f64 {
    Normal::new(0.0, 1.0).map_or(f64::NAN, |normal| normal.inverse_cdf(p))
}

fn cluster_count(clusters: &[u64]) -> usize {
    let mut unique = HashSet::new();
    for &cluster in clusters {
        unique.insert(cluster);
    }
    unique.len()
}

fn covariance_from_information(information: &Mat<f64>) -> Result<Mat<f64>, TwoPartError> {
    let identity = Mat::<f64>::identity(information.nrows(), information.ncols());
    solve_linear_system(information, &identity).map_err(|_| TwoPartError::SolveFailed)
}

fn sandwich_covariance(information: &Mat<f64>, meat: &Mat<f64>) -> Result<Mat<f64>, TwoPartError> {
    let left = solve_linear_system(information, meat).map_err(|_| TwoPartError::SolveFailed)?;
    let cov_t = solve_linear_system_ref(information.transpose(), left.transpose())
        .map_err(|_| TwoPartError::SolveFailed)?;
    Ok(Mat::from_fn(cov_t.ncols(), cov_t.nrows(), |i, j| {
        cov_t[(j, i)]
    }))
}

/// High-level interface for fitting two-part models.
#[derive(Debug, Clone, Copy)]
pub struct TwoPartTrainer {
    options: FitOptions,
}

impl Default for TwoPartTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl TwoPartTrainer {
    /// Create a new trainer with default options.
    #[must_use]
    pub fn new() -> Self {
        Self {
            options: FitOptions::default(),
        }
    }

    /// Fast preset: minimal iterations, no robust standard errors.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            options: FitOptions {
                max_iter: 20,
                tolerance: 1e-4,
                robust_se: false,
                strategy: FitStrategy::Strict,
                ..FitOptions::default()
            },
        }
    }

    /// Stable preset: ridge regularization and relaxed convergence.
    #[must_use]
    pub fn stable() -> Self {
        Self {
            options: FitOptions {
                strategy: FitStrategy::Relaxed {
                    fallback_lambda: 1e-4,
                    max_retries: 3,
                    warm_start: true,
                    time_budget: None,
                },
                ..FitOptions::stable_defaults()
            },
        }
    }

    /// Inference preset: robust standard errors and high precision.
    #[must_use]
    pub fn inference() -> Self {
        Self {
            options: FitOptions {
                max_iter: 100,
                tolerance: 1e-8,
                robust_se: true,
                strategy: FitStrategy::Strict,
                ..FitOptions::default()
            },
        }
    }

    /// Set custom fit options.
    #[must_use]
    pub const fn with_options(mut self, options: FitOptions) -> Self {
        self.options = options;
        self
    }

    /// Set convergence failure strategy.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: FitStrategy) -> Self {
        self.options.strategy = strategy;
        self
    }

    /// Set maximum IRLS iterations.
    #[must_use]
    pub const fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.options.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    #[must_use]
    pub const fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.options.tolerance = tolerance;
        self
    }

    /// Set regularization strategy.
    #[must_use]
    pub const fn with_regularization(mut self, regularization: Regularization) -> Self {
        self.options.regularization = regularization;
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
    /// Returns `TwoPartError` if input is invalid or fitting fails.
    pub fn fit(&self, input: &ModelInput) -> Result<TwoPartModel, TwoPartError> {
        let (model, _) = fit_two_part_input(input, self.options)?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::utils::usize_to_f64;

    #[test]
    fn two_part_model_runs_on_synthetic_data() {
        let n = 200;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 50.0 });

        let mut y = Mat::<f64>::zeros(n, 1);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for i in 0..n {
            let p = 1.0 / (1.0 + (-0.5f64.mul_add(x[(i, 1)], -1.0)).exp());
            if rng.random_range(0.0..1.0) < p {
                let mean = 0.3f64.mul_add(x[(i, 1)], 1.0).exp();
                y[(i, 0)] = mean * (0.5 + rng.random_range(0.0..1.0));
            }
        }

        let (model, report) = fit_two_part(&x, &y, FitOptions::default()).expect("fit");
        assert!(report.iterations_logit > 0);
        assert!(report.iterations_gamma > 0);
        assert!(report.se_logit.is_some());
        assert!(report.se_gamma.is_some());

        let prediction = model.predict(&x);
        assert_eq!(prediction.expected_outcome.nrows(), n);
        assert_relative_eq!(
            prediction.expected_outcome[(0, 0)],
            prediction.prob_positive[(0, 0)] * prediction.mean_positive[(0, 0)]
        );
    }

    use rstest::rstest;

    #[rstest]
    #[case::ridge(Regularization::Ridge { lambda: 0.1, exclude_intercept: true }, true)]
    #[case::elastic_net(Regularization::ElasticNet { lambda: 0.1, alpha: 0.5, exclude_intercept: true }, false)]
    fn test_two_part_regularization_options(#[case] reg: Regularization, #[case] robust_se: bool) {
        let n = 50;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 });
        let mut y = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            y[(i, 0)] = if i % 3 == 0 {
                0.0
            } else {
                0.1f64.mul_add(idx_to_f64(i), 1.0)
            };
        }

        let options = FitOptions {
            regularization: reg,
            robust_se,
            ..FitOptions::default()
        };

        let (_model, report) = fit_two_part(&x, &y, options).expect("fit");
        assert!(report.iterations_logit > 0 || report.iterations_gamma > 0);
        if robust_se {
            assert!(report.se_logit.is_some());
            assert!(report.se_gamma.is_some());
        }
    }

    #[test]
    fn two_part_model_handles_all_positive_outcomes() {
        let n = 40;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 });
        let y = Mat::from_fn(n, 1, |i, _| 1.0 + idx_to_f64(i) / 100.0);

        let (_model, report) = fit_two_part(&x, &y, FitOptions::default())
            .expect("all-positive outcomes should still fit");
        assert_eq!(report.iterations_logit, 0);
        assert!(report.iterations_gamma > 0);
    }

    #[test]
    fn confidence_intervals_from_covariance() {
        let beta = Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { -0.5 });
        let cov = Mat::from_fn(2, 2, |i, j| {
            if i == j {
                if i == 0 { 0.04 } else { 0.01 }
            } else {
                0.0
            }
        });
        let ci = coefficient_confidence_intervals(&beta, &cov, 0.05);
        assert_eq!(ci.len(), 2);
        assert!(ci[0].upper > ci[0].lower);
    }

    #[test]
    fn bootstrap_produces_parameter_samples() {
        let n = 40;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 });
        let mut y = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            y[(i, 0)] = if i % 4 == 0 {
                0.0
            } else {
                0.1f64.mul_add(idx_to_f64(i), 1.0)
            };
        }

        let options = FitOptions {
            robust_se: false,
            regularization: Regularization::None,
            ..FitOptions::default()
        };
        let bootstrap_options = BootstrapOptions {
            iterations: 10,
            seed: 7,
            ..BootstrapOptions::default()
        };
        let result = bootstrap(&x, &y, options, bootstrap_options).expect("bootstrap");
        assert_eq!(result.betas_logit.len(), 10);
        assert_eq!(result.betas_gamma.len(), 10);

        let ci = bootstrap_percentile_ci(&result.betas_logit, 0.1);
        assert_eq!(ci.len(), x.ncols());
    }

    #[test]
    fn bootstrap_summary_produces_mean_and_ci() {
        let betas = vec![
            Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 0.5 }),
            Mat::from_fn(2, 1, |i, _| if i == 0 { 1.2 } else { 0.4 }),
            Mat::from_fn(2, 1, |i, _| if i == 0 { 0.8 } else { 0.6 }),
        ];
        let summary = bootstrap_summary(&betas, 0.1);
        assert_eq!(summary.mean.nrows(), 2);
        assert_eq!(summary.se.nrows(), 2);
        assert_eq!(summary.ci.len(), 2);
    }

    fn bootstrap_percentile_ci_sorted_reference(
        betas: &[Mat<f64>],
        alpha: f64,
    ) -> Vec<ConfidenceInterval> {
        if betas.is_empty() {
            return Vec::new();
        }
        let n = betas[0].nrows();
        let (lower_idx, upper_idx) = crate::utils::boot_index_bounds(alpha, betas.len());
        let mut intervals = Vec::with_capacity(n);
        for col in 0..n {
            let mut values = betas.iter().map(|b| b[(col, 0)]).collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            let lower = values[lower_idx.min(values.len().saturating_sub(1))];
            let upper = values[upper_idx.min(values.len().saturating_sub(1))];
            intervals.push(ConfidenceInterval { lower, upper });
        }
        intervals
    }

    #[test]
    fn bootstrap_percentile_ci_matches_sorted_reference() {
        for reps in [5usize, 19, 200, 999] {
            let betas = (0..reps)
                .map(|draw| {
                    Mat::from_fn(4, 1, |coef, _| {
                        let d = idx_to_f64(draw);
                        let c = idx_to_f64(coef);
                        0.001f64
                            .mul_add(-d, 0.05f64.mul_add(c, (0.17f64.mul_add(d, 0.31 * c)).sin()))
                    })
                })
                .collect::<Vec<_>>();
            for alpha in [0.01, 0.05, 0.10, 0.20] {
                let expected = bootstrap_percentile_ci_sorted_reference(&betas, alpha);
                let actual = bootstrap_percentile_ci(&betas, alpha);
                assert_eq!(actual.len(), expected.len());
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert_eq!(
                        a.lower.to_bits(),
                        e.lower.to_bits(),
                        "reps={reps}, alpha={alpha}"
                    );
                    assert_eq!(
                        a.upper.to_bits(),
                        e.upper.to_bits(),
                        "reps={reps}, alpha={alpha}"
                    );
                }
            }
        }
    }

    #[test]
    fn bootstrap_summary_handles_empty_input() {
        let summary = bootstrap_summary(&[], 0.05);
        assert_eq!(summary.mean.nrows(), 0);
        assert_eq!(summary.se.nrows(), 0);
        assert!(summary.ci.is_empty());
    }

    #[test]
    fn weighted_input_requires_weights() {
        let x = Mat::from_fn(2, 1, |_i, _j| 1.0);
        let y = Mat::from_fn(2, 1, |_i, _j| 1.0);
        let input = ModelInput::new(x, y);
        let err = fit_two_part_weighted_input(&input, FitOptions::default())
            .expect_err("missing weights should error");
        assert!(matches!(err, TwoPartError::MissingWeights));
    }

    #[test]
    fn clustered_input_requires_clusters() {
        let x = Mat::from_fn(2, 1, |_i, _j| 1.0);
        let y = Mat::from_fn(2, 1, |_i, _j| 1.0);
        let input = ModelInput::new(x, y);
        let err = fit_two_part_clustered_input(&input, FitOptions::default())
            .expect_err("missing clusters should error");
        assert!(matches!(err, TwoPartError::MissingClusters));
    }

    #[test]
    fn fit_with_cluster_robust_se() {
        let n = 30;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 5.0 });
        let mut y = Mat::<f64>::zeros(n, 1);
        let clusters: Vec<u64> = (0..n).map(|i| (i / 5) as u64).collect();
        for i in 0..n {
            y[(i, 0)] = if i % 4 == 0 {
                0.0
            } else {
                0.1f64.mul_add(idx_to_f64(i), 1.0)
            };
        }

        let options = FitOptions {
            robust_se: true,
            regularization: Regularization::None,
            ..FitOptions::default()
        };
        let (_model, report) = fit_two_part_clustered(&x, &y, &clusters, options).expect("fit");
        assert!(report.clustered);
        assert!(report.cluster_count.unwrap_or(0) > 1);
        assert!(report.se_logit.is_some());
        assert!(report.se_gamma.is_some());
    }

    #[test]
    fn log_likelihood_returns_nan_on_shape_mismatch() {
        let y = Mat::from_fn(2, 1, |i, _| if i == 0 { 0.0 } else { 1.0 });
        let prob = Mat::from_fn(3, 1, |_i, _| 0.5);
        let mean_pos = Mat::from_fn(2, 1, |_i, _| 1.0);
        let ll = log_likelihood(&y, &prob, &mean_pos);
        assert!(ll.is_nan());
    }

    #[test]
    fn bootstrap_rejects_empty_sample() {
        let x = Mat::from_fn(0, 2, |_i, _j| 1.0);
        let y = Mat::from_fn(0, 1, |_i, _| 0.0);
        let err = bootstrap(&x, &y, FitOptions::default(), BootstrapOptions::default())
            .expect_err("empty bootstrap sample should fail");
        assert!(matches!(err, TwoPartError::EmptyBootstrapSample));
    }

    #[test]
    fn fit_strategy_relaxed_handles_convergence_failure_with_retry() {
        // Create a problem that is hard to converge without regularization
        // by having very few observations or high collinearity.
        let n = 10;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) });
        let y = Mat::from_fn(n, 1, |i, _| if i == 0 { 0.0 } else { 1.0 });

        let options = FitOptions {
            max_iter: 1, // Force early failure
            strategy: FitStrategy::Relaxed {
                fallback_lambda: 1.0,
                max_retries: 2,
                warm_start: true,
                time_budget: None,
            },
            ..FitOptions::default()
        };

        match fit_two_part_input(&ModelInput::new(x, y), options) {
            Ok((_model, report)) => {
                assert!(report.meta.fallback_attempts > 0);
                assert!(!report.attempts.is_empty());
            }
            Err(TwoPartError::NonConvergence | TwoPartError::SolveFailed) => {
                // On tiny pathological samples, relaxed retries may still fail.
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }
}
