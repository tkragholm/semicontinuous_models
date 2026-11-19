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

use epistat_linalg::{Attempt, Factor, Method, ROW_BLOCK, Ridge};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};
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
    epistat_linalg::solve(a_ref, b_ref, &Attempt::SYMMETRIC)
        .map(|solution| solution.x)
        .map_err(|_| TwoPartError::SolveFailed)
}

/// # Errors
///
/// Returns `TwoPartError::SolveFailed` if the solve produces non-finite values.
pub fn solve_linear_system(a: &Mat<f64>, b: &Mat<f64>) -> Result<Mat<f64>, TwoPartError> {
    solve_linear_system_ref(a.as_ref(), b.as_ref())
}

/// Cholesky with a jitter of `1e-10`, growing tenfold up to eight times, on failure,
/// then partial-pivot LU.
const CACHED_FACTOR_ATTEMPTS: [Attempt; 2] = [
    Attempt::ridged(
        Method::Cholesky,
        Ridge {
            start: 1e-10,
            growth: 10.0,
            tries: 8,
        },
    ),
    Attempt::plain(Method::PartialPivLu),
];

/// Factor a (ridged) Gram matrix `X'WX` once for reuse across IRLS iterations whose
/// information matrix does not change, so each pass only back-substitutes the new
/// right-hand side.
///
/// # Errors
///
/// Returns `TwoPartError::SolveFailed` if `matrix` is not square.
pub(crate) fn cached_factor(matrix: &Mat<f64>) -> Result<Factor, TwoPartError> {
    Factor::new(matrix.as_ref(), &CACHED_FACTOR_ATTEMPTS).map_err(|_| TwoPartError::SolveFailed)
}

/// IRLS weights for a log-link GLM whose Fisher-scoring weight is β-independent
/// (gamma / Tweedie power 2): the working weight is exactly the fixed prior
/// (sample/bootstrap) weight, so it is constant across iterations.
///
/// Reproduces the per-iteration computation bit-for-bit. The Tweedie loop forms
/// `base_weight * (μ^(2-power)).max(min_weight)`; at power 2 the variance factor
/// is `1.0` (above `min_weight`), so the weight is precisely `base_weight` with no
/// flooring of the prior weight itself — matched here.
pub(crate) fn fill_constant_irls_weights(
    out: &mut Mat<f64>,
    sample_weights: Option<MatRef<'_, f64>>,
) {
    match sample_weights {
        Some(weights) => out
            .col_as_slice_mut(0)
            .copy_from_slice(col_slice(weights, 0)),
        None => out.col_as_slice_mut(0).fill(1.0),
    }
}

#[must_use]
pub fn mean_column(vector: &Mat<f64>) -> f64 {
    mean_column_ref(vector.as_ref())
}

/// [`mean_column`] on a borrowed column.
#[must_use]
pub fn mean_column_ref(vector: MatRef<'_, f64>) -> f64 {
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

/// The contiguous slice behind column `j` of a column-major view.
///
/// # Panics
///
/// Panics if the view's rows are not contiguous.
#[must_use]
pub(crate) fn col_slice<'a>(matrix: MatRef<'a, f64>, j: usize) -> &'a [f64] {
    matrix
        .col(j)
        .try_as_col_major()
        .expect("column-major view")
        .as_slice()
}

#[must_use]
pub fn matrix_is_finite(matrix: &Mat<f64>) -> bool {
    matrix_ref_is_finite(matrix.as_ref())
}

/// [`matrix_is_finite`] on a borrowed matrix.
#[must_use]
pub fn matrix_ref_is_finite(matrix: MatRef<'_, f64>) -> bool {
    (0..matrix.ncols()).all(|j| col_slice(matrix, j).iter().all(|value| value.is_finite()))
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

/// `X'WX` with the weights floored at zero, exactly symmetric.
#[must_use]
pub fn weighted_xtx(x: &Mat<f64>, weights: &Mat<f64>) -> Mat<f64> {
    weighted_xtx_ref(x.as_ref(), weights.as_ref())
}

/// [`weighted_xtx`] on borrowed matrices.
#[must_use]
pub fn weighted_xtx_ref(x: MatRef<'_, f64>, weights: MatRef<'_, f64>) -> Mat<f64> {
    epistat_linalg::weighted_gram(x, col_slice(weights, 0))
}

/// `dst <- x * beta`, reusing `dst`'s storage so an IRLS loop does not allocate a fresh
/// linear-predictor column every iteration.
pub(crate) fn matvec_into(dst: &mut Mat<f64>, x: MatRef<'_, f64>, beta: &Mat<f64>) {
    epistat_linalg::matvec_into(dst.col_as_slice_mut(0), x, beta.as_ref());
}

/// The mean over the rows of `x` of `row_mean(eta_1, .., eta_K)`, where `eta_k` is the
/// row's linear predictor under `betas[k]`, without forming a column of length `n`.
///
/// The linear predictors are formed block by block with the products and the block
/// partition of [`matvec_into`], so each is bit-identical to the one [`linear_predictor`]
/// gives. The row values are then added in row order, which is the order of
/// [`Iterator::sum`] over the rows. With `weights` the mean is `Σ w_i m_i / Σ w_i` over
/// the rows with `w_i != 0`, accumulated in row order, and without it `Σ m_i / n`.
///
/// # Panics
///
/// Panics if `weights` is given and is not as long as `x` has rows.
pub(crate) fn mean_of_row_values<const K: usize>(
    x: MatRef<'_, f64>,
    betas: [&Mat<f64>; K],
    weights: Option<&[f64]>,
    row_mean: impl Fn([f64; K]) -> f64,
) -> f64 {
    let n = x.nrows();
    if let Some(weights) = weights {
        assert_eq!(weights.len(), n, "one weight per row");
    }
    let mut buffers = [(); K].map(|()| vec![0.0_f64; n.min(ROW_BLOCK)]);
    // `Iterator::sum` starts a float sum at -0.0, the weighted fold at 0.0.
    let mut sum = if weights.is_some() { 0.0 } else { -0.0_f64 };
    let mut total = 0.0_f64;
    let mut start = 0;
    while start < n {
        let len = ROW_BLOCK.min(n - start);
        for (buffer, beta) in buffers.iter_mut().zip(betas) {
            matmul(
                MatMut::from_column_major_slice_mut(&mut buffer[..len], len, 1),
                Accum::Replace,
                x.subrows(start, len),
                beta.as_ref(),
                1.0,
                Par::Seq,
            );
        }
        for offset in 0..len {
            let value = row_mean(std::array::from_fn(|k| buffers[k][offset]));
            match weights {
                None => sum += value,
                Some(weights) => {
                    let weight = weights[start + offset];
                    if weight != 0.0 {
                        sum += weight * value;
                        total += weight;
                    }
                }
            }
        }
        start += len;
    }
    match weights {
        None => sum / n as f64,
        Some(_) => sum / total,
    }
}

/// The linear predictor `x * beta`, computed as in [`matvec_into`].
#[must_use]
pub(crate) fn linear_predictor(x: MatRef<'_, f64>, beta: &Mat<f64>) -> Mat<f64> {
    let mut eta = Mat::<f64>::zeros(x.nrows(), 1);
    matvec_into(&mut eta, x, beta);
    eta
}

#[must_use]
pub fn weighted_xtz(x: &Mat<f64>, weights: &Mat<f64>, response: &Mat<f64>) -> Mat<f64> {
    epistat_linalg::weighted_xtz(
        x.as_ref(),
        col_slice(weights.as_ref(), 0),
        col_slice(response.as_ref(), 0),
    )
}

#[must_use]
pub fn weighted_xtz_with_buffer(
    x: &Mat<f64>,
    weights: &Mat<f64>,
    response: &Mat<f64>,
    weighted_buffer: &mut Vec<f64>,
) -> Mat<f64> {
    weighted_xtz_ref_with_buffer(
        x.as_ref(),
        weights.as_ref(),
        response.as_ref(),
        weighted_buffer,
    )
}

/// [`weighted_xtz_with_buffer`] on borrowed matrices. The products `w_i z_i` are formed
/// in `weighted_buffer`, which an IRLS loop reuses across iterations.
#[must_use]
pub fn weighted_xtz_ref_with_buffer(
    x: MatRef<'_, f64>,
    weights: MatRef<'_, f64>,
    response: MatRef<'_, f64>,
    weighted_buffer: &mut Vec<f64>,
) -> Mat<f64> {
    let n = x.nrows();
    if weighted_buffer.len() < n {
        weighted_buffer.resize(n, 0.0);
    }
    for ((slot, &weight), &value) in weighted_buffer[..n]
        .iter_mut()
        .zip(col_slice(weights, 0))
        .zip(col_slice(response, 0))
    {
        *slot = weight * value;
    }
    epistat_linalg::xtv(x, &weighted_buffer[..n])
}

/// Add a scaled outer-product row contribution to a sandwich "meat" matrix.
pub fn add_row_outer_product_scaled(meat: &mut Mat<f64>, x: &Mat<f64>, row: usize, scale: f64) {
    let p = x.ncols();
    for j in 0..p {
        let scaled_xj = scale * x[(row, j)];
        for k in 0..p {
            meat[(j, k)] = scaled_xj.mul_add(x[(row, k)], meat[(j, k)]);
        }
    }
}

/// Add the scaled outer product `scale * v vᵀ` of a vector into a "meat" matrix.
///
/// Slice-based companion to [`add_row_outer_product_scaled`] for cluster-robust
/// sandwich estimators, where the contribution is the outer product of a
/// per-cluster residual-sum vector rather than a design-matrix row.
pub fn add_outer_product_scaled(meat: &mut Mat<f64>, v: &[f64], scale: f64) {
    let p = v.len();
    for j in 0..p {
        let scaled_vj = scale * v[j];
        for k in 0..p {
            meat[(j, k)] = scaled_vj.mul_add(v[k], meat[(j, k)]);
        }
    }
}

/// The type 7 quantile of sorted values, `None` for no values. `quantile` is clamped to
/// `[0, 1]`.
#[must_use]
pub fn calculate_quantile(sorted_values: &[f64], quantile: f64) -> Option<f64> {
    epistat_stats::quantile::type7(sorted_values, quantile.clamp(0.0, 1.0))
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
            let (predicted_sum, positive_count) =
                rows.iter().fold((0.0, 0usize), |(prob, pos), row| {
                    let is_positive = usize::from(input.outcome[(*row, 0)] > 0.0);
                    (prob + predicted_probability[*row], pos + is_positive)
                });
            let mean_predicted_probability = predicted_sum / usize_to_f64(count);
            let observed_positive_rate = usize_to_f64(positive_count) / usize_to_f64(count);

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
    let mut sorted = values.to_vec();
    summarize_draws_in_place(&mut sorted)
}

/// `summarize_draws` over a buffer the caller owns.
///
/// Computes the mean, then sorts `values` in place for the quantiles (no
/// internal `to_vec`). Use when the draw buffer can be consumed — e.g.
/// per-period effect draws that are summarized exactly once.
#[must_use]
pub fn summarize_draws_in_place(values: &mut [f64]) -> EffectIntervalSummary {
    if values.is_empty() {
        return EffectIntervalSummary {
            mean: f64::NAN,
            q025: f64::NAN,
            q50: f64::NAN,
            q975: f64::NAN,
        };
    }

    let mean = values.iter().sum::<f64>() / usize_to_f64(values.len());
    values.sort_by(f64::total_cmp);

    EffectIntervalSummary {
        mean,
        q025: calculate_quantile(values, 0.025).unwrap_or(f64::NAN),
        q50: calculate_quantile(values, 0.5).unwrap_or(f64::NAN),
        q975: calculate_quantile(values, 0.975).unwrap_or(f64::NAN),
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
