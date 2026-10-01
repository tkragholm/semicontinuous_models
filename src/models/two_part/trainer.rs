//! Preset-based interface for fitting two-part models.

use super::{FitOptions, Regularization, TwoPartError, TwoPartModel, fit_two_part_input};
use crate::input::ModelInput;
use crate::models::FitStrategy;

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
