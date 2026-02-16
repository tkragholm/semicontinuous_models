//! Longitudinal model input containers.
//!
//! This module defines a model-agnostic person-period input used by
//! longitudinal semi-continuous models such as MTP.

use faer::Mat;
use thiserror::Error;

/// Errors returned when validating longitudinal model inputs.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LongitudinalInputError {
    #[error("binary design matrix must have at least one column")]
    EmptyBinaryDesign,
    #[error("mean design matrix must have at least one column")]
    EmptyMeanDesign,
    #[error("outcome must be a single-column matrix")]
    InvalidOutcomeShape,
    #[error(
        "binary design rows ({binary_rows}), mean design rows ({mean_rows}), and outcome rows ({outcome_rows}) must match"
    )]
    DimensionMismatch {
        binary_rows: usize,
        mean_rows: usize,
        outcome_rows: usize,
    },
    #[error("subject id length ({len}) must equal number of rows ({rows})")]
    InvalidSubjectLength { len: usize, rows: usize },
    #[error("time length ({len}) must equal number of rows ({rows})")]
    InvalidTimeLength { len: usize, rows: usize },
    #[error("family id length ({len}) must equal number of rows ({rows})")]
    InvalidFamilyLength { len: usize, rows: usize },
    #[error("input matrices contain non-finite values")]
    NonFiniteMatrix,
    #[error("time contains non-finite values")]
    NonFiniteTime,
    #[error("outcome contains negative values")]
    NegativeOutcome,
}

/// Generic longitudinal input for person-period semi-continuous models.
#[derive(Debug, Clone)]
pub struct LongitudinalModelInput {
    pub outcome: Mat<f64>,
    pub x_binary: Mat<f64>,
    pub x_mean: Mat<f64>,
    pub subject_ids: Vec<u64>,
    pub time: Vec<f64>,
    pub family_ids: Option<Vec<u64>>,
}

impl LongitudinalModelInput {
    #[must_use]
    pub const fn new(
        outcome: Mat<f64>,
        x_binary: Mat<f64>,
        x_mean: Mat<f64>,
        subject_ids: Vec<u64>,
        time: Vec<f64>,
    ) -> Self {
        Self {
            outcome,
            x_binary,
            x_mean,
            subject_ids,
            time,
            family_ids: None,
        }
    }

    #[must_use]
    pub fn with_family_ids(mut self, family_ids: Vec<u64>) -> Self {
        self.family_ids = Some(family_ids);
        self
    }

    /// # Errors
    ///
    /// Returns `LongitudinalInputError` if shapes or values are malformed.
    pub fn validate(&self) -> Result<(), LongitudinalInputError> {
        if self.x_binary.ncols() == 0 {
            return Err(LongitudinalInputError::EmptyBinaryDesign);
        }
        if self.x_mean.ncols() == 0 {
            return Err(LongitudinalInputError::EmptyMeanDesign);
        }
        if self.outcome.ncols() != 1 {
            return Err(LongitudinalInputError::InvalidOutcomeShape);
        }

        let rows = self.outcome.nrows();
        if self.x_binary.nrows() != rows || self.x_mean.nrows() != rows {
            return Err(LongitudinalInputError::DimensionMismatch {
                binary_rows: self.x_binary.nrows(),
                mean_rows: self.x_mean.nrows(),
                outcome_rows: rows,
            });
        }
        if self.subject_ids.len() != rows {
            return Err(LongitudinalInputError::InvalidSubjectLength {
                len: self.subject_ids.len(),
                rows,
            });
        }
        if self.time.len() != rows {
            return Err(LongitudinalInputError::InvalidTimeLength {
                len: self.time.len(),
                rows,
            });
        }
        if let Some(family_ids) = &self.family_ids
            && family_ids.len() != rows
        {
            return Err(LongitudinalInputError::InvalidFamilyLength {
                len: family_ids.len(),
                rows,
            });
        }

        if !matrix_is_finite(&self.outcome)
            || !matrix_is_finite(&self.x_binary)
            || !matrix_is_finite(&self.x_mean)
        {
            return Err(LongitudinalInputError::NonFiniteMatrix);
        }
        if self.time.iter().any(|value| !value.is_finite()) {
            return Err(LongitudinalInputError::NonFiniteTime);
        }
        if (0..rows).any(|i| self.outcome[(i, 0)] < 0.0) {
            return Err(LongitudinalInputError::NegativeOutcome);
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
    fn validate_accepts_well_formed_input() {
        let outcome = Mat::from_fn(3, 1, |_i, _| 1.0);
        let x_binary = Mat::from_fn(3, 2, |_i, j| if j == 0 { 1.0 } else { 0.5 });
        let x_mean = Mat::from_fn(3, 2, |_i, j| if j == 0 { 1.0 } else { 0.25 });
        let subject_ids = vec![10, 10, 11];
        let time = vec![0.0, 1.0, 0.0];
        let input = LongitudinalModelInput::new(outcome, x_binary, x_mean, subject_ids, time);
        assert!(input.validate().is_ok());
    }

    #[test]
    fn validate_rejects_subject_length_mismatch() {
        let outcome = Mat::from_fn(2, 1, |_i, _| 1.0);
        let x_binary = Mat::from_fn(2, 1, |_i, _| 1.0);
        let x_mean = Mat::from_fn(2, 1, |_i, _| 1.0);
        let input = LongitudinalModelInput::new(outcome, x_binary, x_mean, vec![1], vec![0.0, 1.0]);
        let error = input
            .validate()
            .expect_err("subject id mismatch should fail");
        assert!(matches!(
            error,
            LongitudinalInputError::InvalidSubjectLength { .. }
        ));
    }
}
