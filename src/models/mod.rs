use faer::Mat;
use std::time::Duration;

pub mod comparison;
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
