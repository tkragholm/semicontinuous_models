/////////////////////////////////////////////////////////////////////////////////////////////\
//
// Shared linear algebra and statistics utilities for model implementations.
//
// Created on: 24 Jan 2026     Author: Tobias Kragholm
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # Utilities
//!
//! Shared helpers for solving linear systems, computing summary statistics,
//! and working with faer matrices.

use faer::prelude::*;
use faer::{Mat, MatRef, Side};
use num_traits::ToPrimitive;

use crate::input::LongitudinalModelInput;
use crate::models::two_part::TwoPartError;

/// Calibration summary for one predicted-probability bin.
#[derive(Debug, Clone, Copy, Default)]
pub struct CalibrationBinSummary {
    pub bin_index: usize,
    pub lower: f64,
    pub upper: f64,
    pub count: usize,
    pub mean_predicted_probability: f64,
    pub observed_positive_rate: f64,
}

/// Posterior summary interval for a scalar effect quantity.
#[derive(Debug, Clone, Copy, Default)]
pub struct EffectIntervalSummary {
    pub mean: f64,
    pub q025: f64,
    pub q50: f64,
    pub q975: f64,
}

pub fn add_ridge_to_diagonal(matrix: &mut Mat<f64>, lambda: f64, exclude_intercept: bool) {
    if lambda <= 0.0 {
        return;
    }
    let start = usize::from(exclude_intercept);
    let diag_len = matrix.nrows().min(matrix.ncols());
    for idx in start..diag_len {
        matrix[(idx, idx)] += lambda;
    }
}

#[must_use]
pub fn max_abs_diff(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let mut max = 0.0;
    for i in 0..a.nrows() {
        let diff = (a[(i, 0)] - b[(i, 0)]).abs();
        if diff > max {
            max = diff;
        }
    }
    max
}

/// # Errors
///
/// Returns `TwoPartError::SolveFailed` if the solve produces non-finite values.
pub(crate) fn solve_linear_system_ref(
    a_ref: MatRef<'_, f64>,
    b_ref: MatRef<'_, f64>,
) -> Result<Mat<f64>, TwoPartError> {
    if let Ok(cholesky) = a_ref.llt(Side::Lower) {
        let solution = cholesky.solve(b_ref);
        if matrix_is_finite(&solution) {
            return Ok(solution);
        }
    }

    let lblt_solution = a_ref.lblt(Side::Lower).solve(b_ref);
    if matrix_is_finite(&lblt_solution) {
        return Ok(lblt_solution);
    }

    let partial_solution = a_ref.partial_piv_lu().solve(b_ref);
    if matrix_is_finite(&partial_solution) {
        return Ok(partial_solution);
    }

    let robust_solution = a_ref.full_piv_lu().solve(b_ref);
    if !matrix_is_finite(&robust_solution) {
        return Err(TwoPartError::SolveFailed);
    }
    Ok(robust_solution)
}

/// # Errors
///
/// Returns `TwoPartError::SolveFailed` if the solve produces non-finite values.
pub fn solve_linear_system(a: &Mat<f64>, b: &Mat<f64>) -> Result<Mat<f64>, TwoPartError> {
    solve_linear_system_ref(a.as_ref(), b.as_ref())
}

#[must_use]
pub fn mean_column(vector: &Mat<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..vector.nrows() {
        sum += vector[(i, 0)];
    }
    sum / f64::from(u32::try_from(vector.nrows()).unwrap_or(u32::MAX))
}

#[must_use]
/// # Panics
///
/// Panics if `betas` is empty.
pub fn mean_vector(betas: &[Mat<f64>]) -> Mat<f64> {
    assert!(
        !betas.is_empty(),
        "mean_vector requires at least one sample"
    );
    let mut mean = Mat::<f64>::zeros(betas[0].nrows(), 1);
    for beta in betas {
        for i in 0..beta.nrows() {
            mean[(i, 0)] += beta[(i, 0)];
        }
    }
    for i in 0..mean.nrows() {
        mean[(i, 0)] /= f64::from(u32::try_from(betas.len()).unwrap_or(u32::MAX));
    }
    mean
}

#[must_use]
pub fn std_vector(betas: &[Mat<f64>], mean: &Mat<f64>) -> Mat<f64> {
    let mut variance = Mat::<f64>::zeros(mean.nrows(), 1);
    for beta in betas {
        for i in 0..beta.nrows() {
            let diff = beta[(i, 0)] - mean[(i, 0)];
            variance[(i, 0)] = diff.mul_add(diff, variance[(i, 0)]);
        }
    }
    if betas.len() > 1 {
        let denom = f64::from(u32::try_from(betas.len()).unwrap_or(u32::MAX)) - 1.0;
        for i in 0..variance.nrows() {
            variance[(i, 0)] /= denom;
        }
    }
    for i in 0..variance.nrows() {
        variance[(i, 0)] = variance[(i, 0)].max(0.0).sqrt();
    }
    variance
}

#[must_use]
pub fn matrix_is_finite(matrix: &Mat<f64>) -> bool {
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            if !matrix[(i, j)].is_finite() {
                return false;
            }
        }
    }
    true
}

#[must_use]
pub fn to_binary_outcome(outcome: &Mat<f64>) -> Mat<f64> {
    Mat::from_fn(outcome.nrows(), 1, |row, _| {
        if outcome[(row, 0)] > 0.0 { 1.0 } else { 0.0 }
    })
}

#[must_use]
pub fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

#[must_use]
pub fn acceptance_rate(accepted: usize, proposed: usize) -> f64 {
    if proposed == 0 {
        0.0
    } else {
        usize_to_f64(accepted) / usize_to_f64(proposed)
    }
}

#[must_use]
pub const fn retained_draws(iterations: usize, burn_in: usize, thin: usize) -> usize {
    (iterations - burn_in) / thin
}

#[must_use]
pub fn dot_row(matrix: &Mat<f64>, row: usize, coefficients: &[f64]) -> f64 {
    (0..matrix.ncols())
        .map(|col| matrix[(row, col)] * coefficients[col])
        .sum()
}

#[must_use]
pub fn dot_row_mat(matrix: &Mat<f64>, row: usize, coefficients: &Mat<f64>) -> f64 {
    (0..matrix.ncols())
        .map(|col| matrix[(row, col)] * coefficients[(col, 0)])
        .sum()
}

#[must_use]
pub fn quadratic_form_vec(vector: &[f64], matrix: &Mat<f64>) -> f64 {
    let mut quadratic = 0.0;
    for row in 0..vector.len() {
        for col in 0..vector.len() {
            quadratic = (vector[row] * matrix[(row, col)]).mul_add(vector[col], quadratic);
        }
    }
    quadratic
}

#[must_use]
pub fn weighted_xtx(x: &Mat<f64>, weights: &Mat<f64>) -> Mat<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let weighted_x = Mat::from_fn(n, p, |row, col| {
        x[(row, col)] * weights[(row, 0)].max(0.0).sqrt()
    });
    weighted_x.transpose() * &weighted_x
}

#[must_use]
pub fn weighted_xtz(x: &Mat<f64>, weights: &Mat<f64>, response: &Mat<f64>) -> Mat<f64> {
    let mut weighted_buffer = Vec::new();
    weighted_xtz_with_buffer(x, weights, response, &mut weighted_buffer)
}

#[must_use]
pub fn weighted_xtz_with_buffer(
    x: &Mat<f64>,
    weights: &Mat<f64>,
    response: &Mat<f64>,
    weighted_buffer: &mut Vec<f64>,
) -> Mat<f64> {
    let n = x.nrows();
    if weighted_buffer.len() < n {
        weighted_buffer.resize(n, 0.0);
    }
    for row in 0..n {
        weighted_buffer[row] = weights[(row, 0)] * response[(row, 0)];
    }
    let weighted_response = MatRef::from_column_major_slice(&weighted_buffer[..n], n, 1);
    x.transpose() * weighted_response
}

#[must_use]
pub fn calculate_quantile(sorted_values: &[f64], quantile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let last = sorted_values.len() - 1;
    let clamped = quantile.clamp(0.0, 1.0);
    let position = clamped * usize_to_f64(last);
    let lower = position.floor().to_usize().unwrap_or(0);
    let upper = position.ceil().to_usize().unwrap_or(last);

    if lower == upper {
        sorted_values[lower]
    } else {
        let weight = position - position.floor();
        sorted_values[lower].mul_add(1.0 - weight, sorted_values[upper] * weight)
    }
}

#[must_use]
pub fn boot_index_bounds(alpha: f64, n: usize) -> (usize, usize) {
    let n_f = f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    let lower_idx = ((alpha / 2.0) * n_f).floor().to_usize().unwrap_or(0);
    let upper_idx = ((1.0 - alpha / 2.0) * n_f)
        .ceil()
        .to_usize()
        .unwrap_or(0)
        .saturating_sub(1);
    (lower_idx, upper_idx)
}

#[must_use]
pub fn calibration_bins_summary(
    predicted_probability: &[f64],
    input: &LongitudinalModelInput,
    bins: usize,
) -> Vec<CalibrationBinSummary> {
    let mut grouped: Vec<Vec<usize>> = vec![Vec::new(); bins];
    for (row, probability) in predicted_probability.iter().copied().enumerate() {
        let raw = (probability.clamp(0.0, 1.0) * usize_to_f64(bins)).floor();
        let idx = raw.to_usize().unwrap_or(0).min(bins.saturating_sub(1));
        grouped[idx].push(row);
    }

    grouped
        .into_iter()
        .enumerate()
        .map(|(idx, rows)| {
            let lower = usize_to_f64(idx) / usize_to_f64(bins);
            let upper = usize_to_f64(idx + 1) / usize_to_f64(bins);
            if rows.is_empty() {
                return CalibrationBinSummary {
                    bin_index: idx,
                    lower,
                    upper,
                    ..CalibrationBinSummary::default()
                };
            }

            let count = rows.len();
            let mean_predicted_probability = rows
                .iter()
                .map(|row| predicted_probability[*row])
                .sum::<f64>()
                / usize_to_f64(count);
            let observed_positive_rate = rows
                .iter()
                .map(|row| {
                    if input.outcome[(*row, 0)] > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / usize_to_f64(count);

            CalibrationBinSummary {
                bin_index: idx,
                lower,
                upper,
                count,
                mean_predicted_probability,
                observed_positive_rate,
            }
        })
        .collect()
}

#[must_use]
pub fn summarize_draws(values: &[f64]) -> EffectIntervalSummary {
    if values.is_empty() {
        return EffectIntervalSummary::default();
    }

    let mean = values.iter().sum::<f64>() / usize_to_f64(values.len());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    EffectIntervalSummary {
        mean,
        q025: calculate_quantile(&sorted, 0.025),
        q50: calculate_quantile(&sorted, 0.5),
        q975: calculate_quantile(&sorted, 0.975),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn max_abs_diff_matches_expected_value() {
        let a = Mat::from_fn(3, 1, |i, _| f64::from(u32::try_from(i).unwrap_or(u32::MAX)));
        let b = Mat::from_fn(3, 1, |i, _| if i == 2 { 10.0 } else { 0.0 });
        let max = max_abs_diff(&a, &b);
        assert_relative_eq!(max, 8.0);
    }

    #[test]
    fn solve_linear_system_rejects_non_finite_solution() {
        let a = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let b = Mat::from_fn(2, 1, |i, _| if i == 0 { f64::NAN } else { 1.0 });
        let err = solve_linear_system(&a, &b).expect_err("non-finite rhs should fail");
        assert!(matches!(err, TwoPartError::SolveFailed));
    }

    #[test]
    #[should_panic(expected = "mean_vector requires at least one sample")]
    fn mean_vector_panics_on_empty_input() {
        let _ = mean_vector(&[]);
    }

    #[test]
    fn std_vector_is_zero_for_single_sample() {
        let beta = Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { -1.0 });
        let mean = mean_vector(std::slice::from_ref(&beta));
        let std = std_vector(&[beta], &mean);
        assert_relative_eq!(std[(0, 0)], 0.0);
        assert_relative_eq!(std[(1, 0)], 0.0);
    }

    #[test]
    fn matrix_is_finite_detects_nan() {
        let matrix = Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { f64::NAN });
        assert!(!matrix_is_finite(&matrix));
    }

    #[test]
    fn boot_index_bounds_are_ordered_and_in_range() {
        let (lower, upper) = boot_index_bounds(0.1, 100);
        assert!(lower <= upper);
        assert!(upper < 100);
    }

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn boot_index_bounds_invariants(alpha in 0.001..0.5f64, n in 10..1000usize) {
            let (lower, upper) = boot_index_bounds(alpha, n);
            prop_assert!(lower <= upper);
            prop_assert!(upper < n);
        }
    }
}
