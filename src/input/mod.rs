//! # Model inputs
//!
//! Defines a light-weight container for design matrices, outcomes,
//! and optional weights or cluster labels.
//!
//! # Examples
//!
//! ```
//! use faer::Mat;
//! use semicontinuous_models::ModelInput;
//!
//! fn idx_to_f64(idx: usize) -> f64 {
//!     f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
//! }
//!
//! let design_matrix = Mat::from_fn(2, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) });
//! let outcome = Mat::from_fn(2, 1, |i, _| idx_to_f64(i));
//! let input = ModelInput::new(design_matrix, outcome);
//!
//! assert!(input.validate().is_ok());
//! ```
//!
//! ```
//! use faer::Mat;
//! use semicontinuous_models::ModelInput;
//!
//! fn idx_to_f64(idx: usize) -> f64 {
//!     f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
//! }
//!
//! let design_matrix = Mat::from_fn(2, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) });
//! let outcome = Mat::from_fn(3, 1, |i, _| idx_to_f64(i));
//! let input = ModelInput::new(design_matrix, outcome);
//!
//! assert!(input.validate().is_err());
//! ```

use faer::Mat;
use thiserror::Error;

pub mod longitudinal;

pub use longitudinal::{LongitudinalInputError, LongitudinalModelInput};

/// Errors returned when validating model inputs.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum InputError {
    #[error("design matrix must have at least one column")]
    EmptyDesign,
    #[error("outcome must be a single column matrix")]
    InvalidOutcomeShape,
    #[error("design matrix rows ({rows}) must match outcome rows ({len})")]
    DimensionMismatch { rows: usize, len: usize },
    #[error("weights must be a single column matrix with the same number of rows as outcome")]
    InvalidWeightShape,
    #[error("cluster labels length ({labels}) must match outcome rows ({rows})")]
    InvalidClusterLength { labels: usize, rows: usize },
    #[error("design matrix contains non-finite values")]
    NonFiniteDesign,
    #[error("outcome contains non-finite values")]
    NonFiniteOutcome,
    #[error("outcome contains negative values")]
    NegativeOutcome,
    #[error("weights contain non-finite values")]
    NonFiniteWeights,
    #[error("weights must be strictly positive")]
    NonPositiveWeights,
}

#[derive(Debug, Clone)]
pub struct ModelInput {
    pub design_matrix: Mat<f64>,
    pub outcome: Mat<f64>,
    pub sample_weights: Option<Mat<f64>>,
    pub cluster_ids: Option<Vec<u64>>,
}

impl ModelInput {
    #[must_use]
    pub const fn new(design_matrix: Mat<f64>, outcome: Mat<f64>) -> Self {
        Self {
            design_matrix,
            outcome,
            sample_weights: None,
            cluster_ids: None,
        }
    }

    #[must_use]
    pub fn with_sample_weights(mut self, sample_weights: Mat<f64>) -> Self {
        self.sample_weights = Some(sample_weights);
        self
    }

    #[must_use]
    pub fn with_cluster_ids(self, cluster_ids: Vec<u64>) -> Self {
        Self {
            cluster_ids: Some(cluster_ids),
            ..self
        }
    }

    #[must_use]
    pub const fn design_matrix(&self) -> &Mat<f64> {
        &self.design_matrix
    }

    #[must_use]
    pub const fn outcome(&self) -> &Mat<f64> {
        &self.outcome
    }

    #[must_use]
    pub const fn sample_weights(&self) -> Option<&Mat<f64>> {
        self.sample_weights.as_ref()
    }

    #[must_use]
    pub const fn cluster_ids(&self) -> Option<&Vec<u64>> {
        self.cluster_ids.as_ref()
    }

    /// Validate design matrix and outcome only.
    ///
    /// # Errors
    ///
    /// Returns `InputError` if core inputs are malformed.
    pub fn validate_core(&self) -> Result<(), InputError> {
        if self.design_matrix.ncols() == 0 {
            return Err(InputError::EmptyDesign);
        }
        if self.outcome.ncols() != 1 {
            return Err(InputError::InvalidOutcomeShape);
        }
        if self.design_matrix.nrows() != self.outcome.nrows() {
            return Err(InputError::DimensionMismatch {
                rows: self.design_matrix.nrows(),
                len: self.outcome.nrows(),
            });
        }
        if !matrix_is_finite(&self.design_matrix) {
            return Err(InputError::NonFiniteDesign);
        }
        if !matrix_is_finite(&self.outcome) {
            return Err(InputError::NonFiniteOutcome);
        }
        if (0..self.outcome.nrows()).any(|i| self.outcome[(i, 0)] < 0.0) {
            return Err(InputError::NegativeOutcome);
        }
        Ok(())
    }

    /// Validate shapes and values for design matrix, outcome, weights, and clusters.
    ///
    /// # Errors
    ///
    /// Returns `InputError` if inputs are malformed.
    pub fn validate(&self) -> Result<(), InputError> {
        self.validate_core()?;
        if let Some(weights) = &self.sample_weights {
            if weights.ncols() != 1 || weights.nrows() != self.outcome.nrows() {
                return Err(InputError::InvalidWeightShape);
            }
            if !matrix_is_finite(weights) {
                return Err(InputError::NonFiniteWeights);
            }
            if (0..weights.nrows()).any(|i| weights[(i, 0)] <= 0.0) {
                return Err(InputError::NonPositiveWeights);
            }
        }
        if let Some(clusters) = &self.cluster_ids
            && clusters.len() != self.outcome.nrows()
        {
            return Err(InputError::InvalidClusterLength {
                labels: clusters.len(),
                rows: self.outcome.nrows(),
            });
        }
        Ok(())
    }
}

fn matrix_is_finite(matrix: &Mat<f64>) -> bool {
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            if !matrix[(i, j)].is_finite() {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_negative_outcomes() {
        let design_matrix = Mat::from_fn(2, 1, |_i, _j| 1.0);
        let outcome = Mat::from_fn(2, 1, |i, _| if i == 0 { -1.0 } else { 0.0 });
        let input = ModelInput::new(design_matrix, outcome);
        let err = input
            .validate_core()
            .expect_err("negative outcome should fail");
        assert_eq!(err, InputError::NegativeOutcome);
    }

    #[test]
    fn validate_rejects_non_positive_weights() {
        let design_matrix = Mat::from_fn(2, 1, |_i, _j| 1.0);
        let outcome = Mat::from_fn(2, 1, |_i, _| 1.0);
        let w = Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 0.0 });
        let input = ModelInput::new(design_matrix, outcome).with_sample_weights(w);
        let err = input
            .validate()
            .expect_err("non-positive weights should fail");
        assert_eq!(err, InputError::NonPositiveWeights);
    }

    #[test]
    fn validate_rejects_empty_design() {
        let design_matrix = Mat::<f64>::zeros(2, 0);
        let outcome = Mat::from_fn(2, 1, |_i, _| 1.0);
        let input = ModelInput::new(design_matrix, outcome);
        let err = input.validate_core().expect_err("empty design should fail");
        assert_eq!(err, InputError::EmptyDesign);
    }

    #[test]
    fn validate_rejects_invalid_outcome_shape() {
        let design_matrix = Mat::from_fn(2, 1, |_i, _j| 1.0);
        let outcome = Mat::from_fn(2, 2, |_i, _j| 1.0);
        let input = ModelInput::new(design_matrix, outcome);
        let err = input
            .validate_core()
            .expect_err("multi-column outcome should fail");
        assert_eq!(err, InputError::InvalidOutcomeShape);
    }

    #[test]
    fn validate_rejects_dimension_mismatch() {
        let design_matrix = Mat::from_fn(2, 1, |_i, _j| 1.0);
        let outcome = Mat::from_fn(3, 1, |_i, _| 1.0);
        let input = ModelInput::new(design_matrix, outcome);
        let err = input.validate_core().expect_err("row mismatch should fail");
        assert_eq!(err, InputError::DimensionMismatch { rows: 2, len: 3 });
    }

    #[test]
    fn validate_rejects_non_finite_design() {
        let design_matrix = Mat::from_fn(2, 1, |i, _j| if i == 0 { f64::NAN } else { 1.0 });
        let outcome = Mat::from_fn(2, 1, |_i, _| 1.0);
        let input = ModelInput::new(design_matrix, outcome);
        let err = input
            .validate_core()
            .expect_err("non-finite design should fail");
        assert_eq!(err, InputError::NonFiniteDesign);
    }

    #[test]
    fn validate_rejects_non_finite_outcome() {
        let design_matrix = Mat::from_fn(2, 1, |_i, _j| 1.0);
        let outcome = Mat::from_fn(2, 1, |i, _| if i == 0 { f64::INFINITY } else { 1.0 });
        let input = ModelInput::new(design_matrix, outcome);
        let err = input
            .validate_core()
            .expect_err("non-finite outcome should fail");
        assert_eq!(err, InputError::NonFiniteOutcome);
    }

    #[test]
    fn validate_rejects_invalid_weight_shape() {
        let design_matrix = Mat::from_fn(3, 1, |_i, _j| 1.0);
        let outcome = Mat::from_fn(3, 1, |_i, _| 1.0);
        let bad_weights = Mat::from_fn(2, 1, |_i, _| 1.0);
        let input = ModelInput::new(design_matrix, outcome).with_sample_weights(bad_weights);
        let err = input
            .validate()
            .expect_err("invalid weight shape should fail");
        assert_eq!(err, InputError::InvalidWeightShape);
    }

    #[test]
    fn validate_rejects_non_finite_weights() {
        let design_matrix = Mat::from_fn(2, 1, |_i, _j| 1.0);
        let outcome = Mat::from_fn(2, 1, |_i, _| 1.0);
        let bad_weights = Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { f64::NAN });
        let input = ModelInput::new(design_matrix, outcome).with_sample_weights(bad_weights);
        let err = input
            .validate()
            .expect_err("non-finite weights should fail");
        assert_eq!(err, InputError::NonFiniteWeights);
    }

    #[test]
    fn validate_rejects_invalid_cluster_length() {
        let design_matrix = Mat::from_fn(3, 1, |_i, _j| 1.0);
        let outcome = Mat::from_fn(3, 1, |_i, _| 1.0);
        let input = ModelInput::new(design_matrix, outcome).with_cluster_ids(vec![1, 2]);
        let err = input
            .validate()
            .expect_err("cluster length mismatch should fail");
        assert_eq!(err, InputError::InvalidClusterLength { labels: 2, rows: 3 });
    }
}
