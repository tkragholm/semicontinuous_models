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

mod bootstrap;
mod covariance;
mod elastic_net;
mod irls;
mod trainer;

pub use bootstrap::{bootstrap, bootstrap_percentile_ci, bootstrap_summary};
pub use trainer::TwoPartTrainer;

use covariance::{covariance_gamma, covariance_logit};
use irls::{GAMMA_LOG_LINK_ETA_CLAMP, fit_gamma_log_link_weighted, fit_logit_weighted};

use crate::input::{InputError, ModelInput};
use crate::models::covariance::diag_sqrt;
use crate::models::matrix_ops::{select_rows, select_values};
use crate::models::{
    AttemptDiagnostics, FitMetadata, FitStrategy, Model, RetryableError, SolverKind,
    run_with_retries,
};
use crate::utils::linear_predictor;
#[cfg(feature = "bench-internals")]
use crate::utils::weighted_xtz;
use faer::Mat;
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::function::gamma::ln_gamma;
use std::collections::HashSet;
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
    ///
    /// Retained as a SECONDARY criterion. It is absolute and therefore
    /// scale-dependent: a logit whose coefficients sit at 1e3 -- which is what
    /// separation produces -- would need ten significant digits of agreement
    /// between iterates to satisfy it, so on those fits it is unreachable and
    /// the loop always exhausts `max_iter`. `deviance_tolerance` is the primary
    /// test.
    #[builder(default = 1e-6_f64)]
    pub tolerance: f64,
    /// Relative-deviance convergence tolerance: the PRIMARY criterion.
    ///
    /// `|dev - dev_prev| / (|dev| + 0.1) < deviance_tolerance`, which is what
    /// `glm.fit` uses (its `epsilon`, same default). Scale-free, so it behaves
    /// the same on a well-conditioned fit and on one whose coefficients have
    /// drifted large.
    #[builder(default = 1e-8_f64)]
    pub deviance_tolerance: f64,
    /// How many times a step may be halved when it fails to improve the deviance.
    ///
    /// `glm.fit` does the same. Without it the full Newton step is taken
    /// unconditionally, which on a near-separated logit oscillates and burns
    /// every iteration.
    #[builder(default = 25_usize)]
    pub max_step_halvings: usize,
    /// Lower bound on IRLS weights.
    #[builder(default = 1e-6_f64)]
    pub min_weight: f64,
    /// Regularization strategy.
    #[builder(default = Regularization::None)]
    pub regularization: Regularization,
    /// If true, compute sandwich (robust) covariance.
    #[builder(default = false)]
    pub robust_se: bool,
    /// If true (the default), compute the model-based coefficient covariance / SEs
    /// during the fit. A clustered bootstrap draws hundreds–thousands of replicate
    /// fits and reads ONLY the coefficients (the CI comes from the spread of betas
    /// across replicates), so it sets this false to skip the per-replicate covariance
    /// — two `X'WX` builds and two p×p solves that would be computed and discarded.
    #[builder(default = true)]
    pub compute_covariance: bool,
    /// Strategy for handling non-convergence.
    #[builder(default = FitStrategy::Strict)]
    pub strategy: FitStrategy,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tolerance: 1e-6,
            deviance_tolerance: 1e-8,
            max_step_halvings: 25,
            min_weight: 1e-6,
            regularization: Regularization::None,
            robust_se: false,
            compute_covariance: true,
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
        let eta_logit = linear_predictor(x, &self.beta_logit);
        let eta_gamma = linear_predictor(x, &self.beta_gamma);
        for i in 0..x.nrows() {
            let prob = 1.0 / (1.0 + (-eta_logit[(i, 0)]).exp());
            let mean = eta_gamma[(i, 0)]
                .clamp(-GAMMA_LOG_LINK_ETA_CLAMP, GAMMA_LOG_LINK_ETA_CLAMP)
                .exp();
            out.prob_positive[(i, 0)] = prob;
            out.mean_positive[(i, 0)] = mean;
            out.expected_outcome[(i, 0)] = prob * mean;
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
    fit_two_part_weighted(x, y, &weights, options, None)
}

/// Full-sample coefficients used to warm-start a bootstrap replicate's two-part IRLS.
///
/// The replicate design has the same covariate columns as the full sample, so the
/// full-sample logit and gamma coefficients seed the replicate's two IRLS loops; they
/// converge to the same optimum in far fewer iterations. Ignored if the dimensions do
/// not match (the leaf fitters fall back to their cold start).
#[derive(Clone, Debug)]
pub struct TwoPartWarmStart {
    pub beta_logit: Mat<f64>,
    pub beta_gamma: Mat<f64>,
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
    fit_two_part_input_warm(input, options, None)
}

/// Fit a two-part model from a `ModelInput`, warm-starting the IRLS from `warm`.
///
/// A bootstrap replicate is a perturbation of the full sample, so seeding both the logit
/// and gamma IRLS loops with the full-sample coefficients converges in far fewer
/// iterations to the same optimum. The warm start is applied on the weighted paths (the
/// bootstrap always carries weights); the unweighted paths cold-start.
///
/// # Errors
///
/// Returns `TwoPartError` if inputs are malformed or the solver fails to converge.
pub fn fit_two_part_input_warm(
    input: &ModelInput,
    options: FitOptions,
    warm: Option<&TwoPartWarmStart>,
) -> Result<(TwoPartModel, TwoPartReport), TwoPartError> {
    input.validate()?;

    let start_time = Instant::now();
    let retried = run_with_retries(
        options,
        options.strategy,
        start_time,
        TwoPartError::NonConvergence,
        |base, fallback_lambda, scale| FitOptions {
            regularization: Regularization::Ridge {
                lambda: fallback_lambda * scale,
                exclude_intercept: true,
            },
            ..base
        },
        |current| match current.regularization {
            Regularization::Ridge { lambda, .. } | Regularization::ElasticNet { lambda, .. } => {
                lambda
            }
            Regularization::BayesianRidge { prior_scale, .. } => 1.0 / (prior_scale * prior_scale),
            Regularization::None => 0.0,
        },
        |current_options| match (&input.sample_weights, &input.cluster_ids) {
            (Some(weights), Some(clusters)) => fit_two_part_clustered_weighted(
                &input.design_matrix,
                &input.outcome,
                weights,
                clusters,
                current_options,
                warm,
            ),
            (Some(weights), None) => fit_two_part_weighted(
                &input.design_matrix,
                &input.outcome,
                weights,
                current_options,
                warm,
            ),
            (None, Some(clusters)) => fit_two_part_clustered(
                &input.design_matrix,
                &input.outcome,
                clusters,
                current_options,
            ),
            (None, None) => fit_two_part(&input.design_matrix, &input.outcome, current_options),
        },
    )?;

    let (mut model, mut report) = retried.fit;
    report.attempts = retried.attempts;
    report.meta.fallback_attempts = retried.attempt_idx;
    model.report = report.clone();
    Ok((model, report))
}

impl RetryableError for TwoPartError {
    fn is_retryable(&self) -> bool {
        matches!(self, Self::NonConvergence | Self::SolveFailed)
    }
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
    warm: Option<&TwoPartWarmStart>,
) -> Result<(TwoPartModel, TwoPartReport), TwoPartError> {
    let start_time = Instant::now();

    let is_positive = Mat::from_fn(y.nrows(), 1, |i, _| if y[(i, 0)] > 0.0 { 1.0 } else { 0.0 });
    let (beta_logit, iterations_logit) = fit_logit_weighted(
        x,
        &is_positive,
        weights,
        options,
        warm.map(|w| &w.beta_logit),
    )?;

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
        fit_gamma_log_link_weighted(&x_pos, &y_pos, &w_pos, options, warm.map(|w| &w.beta_gamma))?;

    // The model-based covariance is computed only when requested. The clustered
    // bootstrap disables it (it reads only the coefficients), and the cluster-robust
    // path recomputes a sandwich covariance over this anyway — so for both, these two
    // `X'WX` builds + p×p solves would be pure waste. `None` here keeps the report's
    // se/cov optional fields exactly as an "SEs not computed" fit already reports them.
    let (cov_logit, cov_gamma) = if options.compute_covariance {
        (
            Some(covariance_logit(
                x,
                &is_positive,
                weights,
                &beta_logit,
                None,
                options,
            )?),
            Some(covariance_gamma(
                &x_pos,
                &y_pos,
                &w_pos,
                &beta_gamma,
                None,
                options,
            )?),
        )
    } else {
        (None, None)
    };
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
    fit_two_part_clustered_weighted(x, y, &weights, clusters, options, None)
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
    warm: Option<&TwoPartWarmStart>,
) -> Result<(TwoPartModel, TwoPartReport), TwoPartError> {
    let (model, base_report) = fit_two_part_weighted(x, y, weights, options, warm)?;
    if !options.robust_se {
        return Ok((model, base_report));
    }

    let is_positive = Mat::from_fn(y.nrows(), 1, |i, _| if y[(i, 0)] > 0.0 { 1.0 } else { 0.0 });
    let positive_indices: Vec<usize> = (0..y.nrows()).filter(|&idx| y[(idx, 0)] > 0.0).collect();
    let x_pos = select_rows(x, &positive_indices);
    let y_pos = select_values(y, &positive_indices);
    let w_pos = select_values(weights, &positive_indices);
    let clusters_pos = select_cluster_ids(clusters, &positive_indices);

    let cov_logit = covariance_logit(
        x,
        &is_positive,
        weights,
        &model.beta_logit,
        Some(clusters),
        options,
    )?;
    let cov_gamma = covariance_gamma(
        &x_pos,
        &y_pos,
        &w_pos,
        &model.beta_gamma,
        Some(&clusters_pos),
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

fn select_cluster_ids(clusters: &[u64], indices: &[usize]) -> Vec<u64> {
    indices.iter().map(|idx| clusters[*idx]).collect()
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
    clusters.iter().collect::<HashSet<_>>().len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::usize_to_f64;
    use approx::assert_relative_eq;
    use rand::prelude::*;

    #[test]
    fn two_part_model_runs_on_synthetic_data() {
        let n = 200;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 50.0 },
        );

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
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 },
        );
        let mut y = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            y[(i, 0)] = if i % 3 == 0 {
                0.0
            } else {
                0.1f64.mul_add(usize_to_f64(i), 1.0)
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
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 },
        );
        let y = Mat::from_fn(n, 1, |i, _| 1.0 + usize_to_f64(i) / 100.0);

        let (_model, report) = fit_two_part(&x, &y, FitOptions::default())
            .expect("all-positive outcomes should still fit");
        assert_eq!(report.iterations_logit, 0);
        assert!(report.iterations_gamma > 0);
    }

    #[test]
    fn gamma_fit_survives_heavy_tailed_outcome() {
        // A mass of small positive values with a few extreme outliers. Before
        // clamping the log-link predictor, the IRLS could drive mu = exp(eta) to
        // +inf and diverge on this shape; it must now fit to finite coefficients.
        let n = 80;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                usize_to_f64(i) / usize_to_f64(n)
            }
        });
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 16 == 0 {
                5.0e7
            } else {
                40.0 + usize_to_f64(i)
            }
        });
        let w = Mat::from_fn(n, 1, |_, _| 1.0);

        let (beta, iterations) =
            fit_gamma_log_link_weighted(&x, &y, &w, FitOptions::stable_defaults(), None)
                .expect("heavy-tailed gamma fit should converge");
        assert!(iterations >= 1);
        assert!(
            crate::utils::matrix_is_finite(&beta),
            "estimated coefficients must be finite"
        );
    }

    #[test]
    fn gamma_prediction_clamps_extreme_linear_predictor() {
        // Fit a benign model, then predict at an extreme covariate. Without the
        // clamp, exp(eta) overflows to +inf; the clamp keeps the predicted mean
        // (and expected outcome) finite.
        let n = 30;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| {
                if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 }
            },
        );
        let y = Mat::from_fn(n, 1, |i, _| 10.0 + usize_to_f64(i));
        let (model, _report) =
            fit_two_part(&x, &y, FitOptions::stable_defaults()).expect("benign fit");

        let x_extreme = Mat::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 1.0e6 });
        let prediction = model.predict(&x_extreme);
        assert!(
            prediction.mean_positive[(0, 0)].is_finite(),
            "clamp must keep exp(eta) finite at extreme covariates"
        );
        assert!(prediction.expected_outcome[(0, 0)].is_finite());
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
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 },
        );
        let mut y = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            y[(i, 0)] = if i % 4 == 0 {
                0.0
            } else {
                0.1f64.mul_add(usize_to_f64(i), 1.0)
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
                        let d = usize_to_f64(draw);
                        let c = usize_to_f64(coef);
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
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 5.0 },
        );
        let mut y = Mat::<f64>::zeros(n, 1);
        let clusters: Vec<u64> = (0..n).map(|i| (i / 5) as u64).collect();
        for i in 0..n {
            y[(i, 0)] = if i % 4 == 0 {
                0.0
            } else {
                0.1f64.mul_add(usize_to_f64(i), 1.0)
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
    fn compute_covariance_false_keeps_coefficients_and_drops_ses() {
        // The bootstrap-replicate path sets compute_covariance=false to skip the
        // model-based covariance it never reads. That MUST leave the fitted
        // coefficients byte-identical (the bootstrap CI is the spread of these
        // betas) and only suppress the unused se/cov fields.
        let n = 60;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 20.0 },
        );
        // A two-part shape: a zero spike plus positive values.
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 3 == 0 {
                0.0
            } else {
                1.0 + usize_to_f64(i)
            }
        });

        let with_cov = FitOptions::default();
        let without_cov = FitOptions {
            compute_covariance: false,
            ..FitOptions::default()
        };

        let (model_a, report_a) = fit_two_part(&x, &y, with_cov).expect("fit with covariance");
        let (model_b, report_b) =
            fit_two_part(&x, &y, without_cov).expect("fit without covariance");

        // Coefficients are identical — the optimization only removes discarded work.
        for j in 0..model_a.beta_logit.nrows() {
            assert_eq!(model_a.beta_logit[(j, 0)], model_b.beta_logit[(j, 0)]);
        }
        for j in 0..model_a.beta_gamma.nrows() {
            assert_eq!(model_a.beta_gamma[(j, 0)], model_b.beta_gamma[(j, 0)]);
        }

        // Default fit reports SEs; the covariance-free fit reports none.
        assert!(report_a.se_logit.is_some() && report_a.se_gamma.is_some());
        assert!(report_b.se_logit.is_none() && report_b.se_gamma.is_none());
        assert!(report_b.cov_logit.is_none() && report_b.cov_gamma.is_none());
    }

    #[test]
    fn two_part_warm_start_matches_cold_start() {
        // A weighted input hits the warm-startable path (the bootstrap carries weights).
        let n = 80;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 20.0 },
        );
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 3 == 0 {
                0.0
            } else {
                1.0 + usize_to_f64(i)
            }
        });
        let input = ModelInput::new(x, y).with_sample_weights(Mat::from_fn(n, 1, |_, _| 1.0));
        let options = FitOptions::default();

        let (cold, _) = fit_two_part_input(&input, options).expect("cold fit");
        let warm_start = TwoPartWarmStart {
            beta_logit: cold.beta_logit.clone(),
            beta_gamma: cold.beta_gamma.clone(),
        };
        let (warm, _) =
            fit_two_part_input_warm(&input, options, Some(&warm_start)).expect("warm fit");

        // Both IRLS loops converge to the same optimum (to the step-convergence
        // tolerance — see the Tweedie warm-start test for the precision rationale).
        for i in 0..cold.beta_logit.nrows() {
            assert!((cold.beta_logit[(i, 0)] - warm.beta_logit[(i, 0)]).abs() < 1e-3);
        }
        for i in 0..cold.beta_gamma.nrows() {
            assert!((cold.beta_gamma[(i, 0)] - warm.beta_gamma[(i, 0)]).abs() < 1e-3);
        }
        // Warm-starting from the converged fit converges in no more iterations.
        assert!(warm.report.iterations_logit <= cold.report.iterations_logit);
        assert!(warm.report.iterations_gamma <= cold.report.iterations_gamma);
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
        // A problem that is hard to converge without regularization, via high
        // collinearity and a hard iteration cap.
        //
        // This used to put a SINGLE zero against two parameters, which is not a
        // hard fit -- it is a separated one, where the logit MLE is at infinity.
        // The separation guard now returns the intercept-only prevalence model
        // for that shape in zero iterations, so there was nothing left to retry
        // and this test failed on `fallback_attempts > 0`. The minority class is
        // now larger than the parameter count, which keeps the logit identified
        // and leaves `max_iter: 1` as the thing that forces the retry.
        let n = 10;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) });
        let y = Mat::from_fn(n, 1, |i, _| if i % 3 == 0 { 0.0 } else { 1.0 });

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
