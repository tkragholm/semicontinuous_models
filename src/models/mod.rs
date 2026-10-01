use faer::Mat;
use std::time::Duration;

pub mod comparison;
pub(crate) mod covariance;
pub mod lognormal;
pub mod matrix_ops;
pub mod mtp;
pub mod selection;
pub mod tweedie;
pub mod two_part;

/// Finalize a successful retryable fit by updating metadata and returning the model/report pair.
#[macro_export]
macro_rules! finalize_retry_fit {
    ($model:ident, $report:ident, $attempts:ident, $start_time:ident, $attempt_idx:ident) => {{
        let execution_time = $start_time.elapsed();
        $report.meta.execution_time = execution_time;
        $report.meta.fallback_attempts = $attempt_idx;
        $report.attempts = $attempts;
        $model.report = $report.clone();
        return Ok(($model, $report));
    }};
}

/// Unified interface for all semicontinuous models.
pub trait Model {
    /// Prediction output type (e.g., `TwoPartPrediction`).
    type Prediction;
    /// Diagnostic report type (e.g., `TwoPartReport`).
    type Report;

    /// Generate predictions from a design matrix (allocates a new Prediction).
    fn predict(&self, x: &Mat<f64>) -> Self::Prediction;

    /// Generate predictions into a pre-allocated buffer (zero-allocation hot path).
    /// Panics if `out` dimensions do not match the expected output shape.
    fn predict_into(&self, x: &Mat<f64>, out: &mut Self::Prediction);

    /// Access model diagnostics and fit metadata.
    fn report(&self) -> &Self::Report;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    Irls,
    LBfgs,
    NewtonRaphson,
    Mcmc,
}

/// Standardized metadata for all model fits.
#[derive(Debug, Clone)]
pub struct FitMetadata {
    pub iterations: usize,
    pub converged: bool,
    pub execution_time: Duration,
    pub solver: SolverKind,
    pub gradient_evaluations: usize,
    pub line_search_steps: usize,
    pub factorization_count: usize,
    pub fallback_attempts: usize,
}

impl Default for FitMetadata {
    fn default() -> Self {
        Self {
            iterations: 0,
            converged: false,
            execution_time: Duration::default(),
            solver: SolverKind::Irls,
            gradient_evaluations: 0,
            line_search_steps: 0,
            factorization_count: 0,
            fallback_attempts: 0,
        }
    }
}

/// Strategy for handling convergence failures.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FitStrategy {
    /// Fail immediately on non-convergence.
    #[default]
    Strict,
    /// Retry with progressively stricter regularization if primary fit fails.
    Relaxed {
        /// Starting lambda for fallback regularization.
        fallback_lambda: f64,
        /// Maximum number of retry attempts.
        max_retries: usize,
        /// Warm-start retries from the previous attempt's coefficients.
        warm_start: bool,
        /// Maximum total wall-clock time across all attempts.
        time_budget: Option<Duration>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptOutcome {
    Converged,
    Diverged,
    TimedOut,
    EarlyAbort,
}

/// Diagnostics for each attempt in a multi-retry fit.
#[derive(Debug, Clone)]
pub struct AttemptDiagnostics {
    pub attempt: usize,
    pub lambda_used: f64,
    pub meta: FitMetadata,
    pub outcome: AttemptOutcome,
}

/// An error a fit may be retried after under [`FitStrategy::Relaxed`].
pub(crate) trait RetryableError {
    /// True for non-convergence and failed solves, which stronger regularisation can fix.
    fn is_retryable(&self) -> bool;
}

/// A successful fit from [`run_with_retries`]: the fitted value, the failed attempts
/// before it, and the index of the attempt that succeeded.
pub(crate) struct RetriedFit<T> {
    pub fit: T,
    pub attempts: Vec<AttemptDiagnostics>,
    pub attempt_idx: usize,
}

/// Run `fit` once under [`FitStrategy::Strict`], or up to `1 + max_retries` times under
/// [`FitStrategy::Relaxed`].
///
/// Attempt `k > 0` fits with `relax(base, fallback_lambda, 10^(k-1))`. A retryable
/// failure is recorded as `Diverged` with `lambda_used(options)` and the next attempt
/// runs. Any other error is returned at once. When the time budget is spent before an
/// attempt, a `TimedOut` record is added and no further attempt runs. If every attempt
/// fails, the last retryable error is returned, or `initial_err` if none ran.
pub(crate) fn run_with_retries<O: Copy, T, E: RetryableError>(
    base: O,
    strategy: FitStrategy,
    start_time: std::time::Instant,
    initial_err: E,
    relax: impl Fn(O, f64, f64) -> O,
    lambda_used: impl Fn(&O) -> f64,
    mut fit: impl FnMut(O) -> Result<T, E>,
) -> Result<RetriedFit<T>, E> {
    let max_attempts = match strategy {
        FitStrategy::Strict => 1,
        FitStrategy::Relaxed { max_retries, .. } => 1 + max_retries,
    };
    let mut current = base;
    let mut attempts = Vec::new();
    let mut last_err = initial_err;

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
            let scale = 10.0f64.powi(i32::try_from(attempt_idx - 1).unwrap_or(0));
            current = relax(base, fallback_lambda, scale);
        }

        let lambda = lambda_used(&current);
        match fit(current) {
            Ok(fit) => {
                return Ok(RetriedFit {
                    fit,
                    attempts,
                    attempt_idx,
                });
            }
            Err(err) if err.is_retryable() => {
                attempts.push(AttemptDiagnostics {
                    attempt: attempt_idx,
                    lambda_used: lambda,
                    meta: FitMetadata {
                        converged: false,
                        execution_time: start_time.elapsed(),
                        ..FitMetadata::default()
                    },
                    outcome: AttemptOutcome::Diverged,
                });
                last_err = err;
            }
            Err(err) => return Err(err),
        }
    }

    Err(last_err)
}
