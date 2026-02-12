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

use faer::Mat;
use faer::prelude::Solve;
use num_traits::ToPrimitive;

use crate::models::two_part::TwoPartError;

#[must_use]
pub fn diag_from_vec(weights: &Mat<f64>) -> Mat<f64> {
    Mat::from_fn(weights.nrows(), weights.nrows(), |i, j| {
        if i == j { weights[(i, 0)] } else { 0.0 }
    })
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
pub fn solve_linear_system(a: &Mat<f64>, b: &Mat<f64>) -> Result<Mat<f64>, TwoPartError> {
    let rhs = b.clone();
    let lu = a.full_piv_lu();
    let solution = lu.solve(rhs);
    if !matrix_is_finite(&solution) {
        return Err(TwoPartError::SolveFailed);
    }
    Ok(solution)
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
            variance[(i, 0)] += diff * diff;
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn diag_from_vec_places_weights_on_diagonal() {
        let weights = Mat::from_fn(3, 1, |i, _| {
            f64::from(u32::try_from(i + 1).unwrap_or(u32::MAX))
        });
        let diag = diag_from_vec(&weights);
        assert_relative_eq!(diag[(0, 0)], 1.0);
        assert_relative_eq!(diag[(1, 1)], 2.0);
        assert_relative_eq!(diag[(2, 2)], 3.0);
        assert_relative_eq!(diag[(0, 1)], 0.0);
    }

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
}
