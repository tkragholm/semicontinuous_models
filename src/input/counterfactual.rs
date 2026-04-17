use faer::Mat;
use thiserror::Error;

use crate::input::ModelInput;
use crate::preprocess::{column_has_variation, drop_constant_columns};

/// Default threshold where two-part models are treated as not applicable.
pub const TWO_PART_NOT_APPLICABLE_POSITIVE_SHARE: f64 = 0.995;

/// Generic prepared input and counterfactual design matrices.
#[derive(Debug, Clone)]
pub struct PreparedCounterfactualInput {
    /// Validated model input.
    pub input: ModelInput,
    /// Design matrix with exposure set to 1.
    pub x_treated: Mat<f64>,
    /// Design matrix with exposure set to 0.
    pub x_control: Mat<f64>,
    /// Number of rows used to build the input.
    pub n_rows: usize,
    /// Number of strictly positive outcome values.
    pub n_positive: usize,
}

/// Input-build errors for counterfactual model preparation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum CounterfactualInputBuildError {
    #[error("no rows")]
    NoRows,
    #[error("exposure column index is out of bounds for design matrix")]
    InvalidExposureColumn,
    #[error("non-finite outcome values")]
    NonFiniteOutcome,
    #[error("insufficient exposure variation")]
    InsufficientExposureVariation,
    #[error("exposure column dropped unexpectedly")]
    ExposureColumnDropped,
}

impl CounterfactualInputBuildError {
    /// Return a stable, human-readable message.
    #[must_use]
    pub const fn message(&self) -> &'static str {
        match self {
            Self::NoRows => "no rows",
            Self::InvalidExposureColumn => {
                "exposure column index is out of bounds for design matrix"
            }
            Self::NonFiniteOutcome => "non-finite outcome values",
            Self::InsufficientExposureVariation => "insufficient exposure variation",
            Self::ExposureColumnDropped => "exposure column dropped unexpectedly",
        }
    }
}

/// Clamp finite outcomes to non-negative values for semi-continuous modeling.
#[must_use]
pub const fn sanitize_model_outcome(raw: f64) -> f64 {
    if raw.is_finite() { raw.max(0.0) } else { raw }
}

/// Return whether two-part models are applicable for a positive outcome share.
#[must_use]
pub fn two_part_applicability(positive_share: f64) -> bool {
    positive_share < TWO_PART_NOT_APPLICABLE_POSITIVE_SHARE
}

/// Build reusable model input with treated/control counterfactual matrices.
///
/// `design_value` must return the observed design matrix element for `column`.
/// The `exposure_column` values are toggled to 1/0 in the returned counterfactual
/// matrices after constant-column dropping.
///
/// # Errors
///
/// Returns `CounterfactualInputBuildError` when rows are empty, exposure has no
/// variation, outcomes are non-finite, or exposure metadata is invalid.
pub fn build_counterfactual_input<'a, R: 'a, FRow, FOutcome, FDesign>(
    n_rows: usize,
    mut row_at: FRow,
    outcome: FOutcome,
    design_columns: usize,
    exposure_column: usize,
    design_value: FDesign,
    constant_column_tolerance: f64,
) -> Result<PreparedCounterfactualInput, CounterfactualInputBuildError>
where
    FRow: FnMut(usize) -> &'a R,
    FOutcome: Fn(&R) -> f64,
    FDesign: Fn(&R, usize) -> f64,
{
    if n_rows == 0 {
        return Err(CounterfactualInputBuildError::NoRows);
    }
    if exposure_column >= design_columns {
        return Err(CounterfactualInputBuildError::InvalidExposureColumn);
    }

    let mut y_values = Vec::with_capacity(n_rows);
    let mut n_positive = 0_usize;
    for row_idx in 0..n_rows {
        let row = row_at(row_idx);
        let y = sanitize_model_outcome(outcome(row));
        if !y.is_finite() {
            return Err(CounterfactualInputBuildError::NonFiniteOutcome);
        }
        if y > 0.0 {
            n_positive += 1;
        }
        y_values.push(y);
    }

    let x_full = Mat::from_fn(n_rows, design_columns, |row_idx, col_idx| {
        let row = row_at(row_idx);
        design_value(row, col_idx)
    });

    if !column_has_variation(&x_full, exposure_column, constant_column_tolerance) {
        return Err(CounterfactualInputBuildError::InsufficientExposureVariation);
    }

    let (x, kept_columns) =
        drop_constant_columns(&x_full, constant_column_tolerance, &[0, exposure_column]);
    let Some(exposure_col_reindexed) = kept_columns
        .iter()
        .position(|column| *column == exposure_column)
    else {
        return Err(CounterfactualInputBuildError::ExposureColumnDropped);
    };

    let y = Mat::from_fn(n_rows, 1, |row_idx, _| y_values[row_idx]);

    let mut x_treated = x.clone();
    let mut x_control = x.clone();
    for row_idx in 0..n_rows {
        x_treated[(row_idx, exposure_col_reindexed)] = 1.0;
        x_control[(row_idx, exposure_col_reindexed)] = 0.0;
    }

    Ok(PreparedCounterfactualInput {
        input: ModelInput::new(x, y),
        x_treated,
        x_control,
        n_rows,
        n_positive,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CounterfactualInputBuildError, build_counterfactual_input, two_part_applicability,
    };

    #[derive(Debug, Clone, Copy)]
    struct Row {
        outcome: f64,
        exposed: bool,
        followup: i32,
        male: bool,
    }

    fn design_value(row: &Row, column: usize) -> f64 {
        match column {
            0 => 1.0,
            1 if row.exposed => 1.0,
            2 => f64::from(row.followup),
            3 if row.male => 1.0,
            _ => 0.0,
        }
    }

    #[test]
    fn build_counterfactual_input_rejects_empty_rows() {
        let rows: [Row; 0] = [];
        let result = build_counterfactual_input(
            rows.len(),
            |idx| &rows[idx],
            |row| row.outcome,
            4,
            1,
            design_value,
            1e-12,
        );
        assert!(matches!(result, Err(CounterfactualInputBuildError::NoRows)));
    }

    #[test]
    fn build_counterfactual_input_rejects_non_finite_outcomes() {
        let rows = [Row {
            outcome: f64::NAN,
            exposed: true,
            followup: 0,
            male: false,
        }];
        let result = build_counterfactual_input(
            rows.len(),
            |idx| &rows[idx],
            |row| row.outcome,
            4,
            1,
            design_value,
            1e-12,
        );
        assert!(matches!(
            result,
            Err(CounterfactualInputBuildError::NonFiniteOutcome)
        ));
    }

    #[test]
    fn build_counterfactual_input_constructs_counterfactual_matrices() {
        let rows = [
            Row {
                outcome: 0.0,
                exposed: false,
                followup: 0,
                male: false,
            },
            Row {
                outcome: 10.0,
                exposed: true,
                followup: 1,
                male: true,
            },
        ];
        let prepared = build_counterfactual_input(
            rows.len(),
            |idx| &rows[idx],
            |row| row.outcome,
            4,
            1,
            design_value,
            1e-12,
        )
        .expect("valid prepared input");
        assert_eq!(prepared.n_rows, 2);
        assert_eq!(prepared.n_positive, 1);
        assert!((prepared.x_treated[(0, 1)] - 1.0).abs() < 1e-12);
        assert!((prepared.x_control[(0, 1)] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn two_part_applicability_respects_default_threshold() {
        assert!(two_part_applicability(0.5));
        assert!(!two_part_applicability(0.995));
    }
}
