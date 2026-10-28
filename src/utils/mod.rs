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

use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{self, BlockStructure};
use faer::linalg::solvers::{Llt, PartialPivLu};
use faer::prelude::*;
use faer::{Accum, Mat, MatMut, MatRef, Par, Side};
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::cell::Cell;
use std::ops::Range;
use std::sync::OnceLock;

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

thread_local! {
    static SERIAL_INNER: Cell<bool> = const { Cell::new(false) };
}

/// Marks the current thread's fit as serial for the lifetime of the guard: every
/// row-block pass, matrix-vector product and dense solve run by the fit stays on this
/// thread. Guards nest, and a serial scope is never widened by an inner one.
pub(crate) struct SerialInnerScope {
    previous: bool,
}

impl SerialInnerScope {
    /// Enter a serial scope when `serial` is true, and leave the current mode alone
    /// otherwise.
    pub(crate) fn enter(serial: bool) -> Self {
        let previous = SERIAL_INNER.with(Cell::get);
        if serial {
            SERIAL_INNER.with(|flag| flag.set(true));
        }
        Self { previous }
    }
}

impl Drop for SerialInnerScope {
    fn drop(&mut self) {
        SERIAL_INNER.with(|flag| flag.set(self.previous));
    }
}

fn serial_inner() -> bool {
    SERIAL_INNER.with(Cell::get)
}

/// Run `f` where faer's global parallelism resolves to one thread: on a rayon pool of
/// one worker, `Par::rayon(0)` is the size of that pool.
fn on_single_thread<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "semicontinuous-serial".to_owned())
            .build()
            .expect("single-thread pool")
    })
    .install(f)
}

/// Cholesky factor of the lower triangle of `a`, computed with `Par::Seq`.
fn llt_factor_seq(a: MatRef<'_, f64>) -> Option<Mat<f64>> {
    let n = a.nrows();
    let mut lower = Mat::<f64>::zeros(n, n);
    lower.copy_from_triangular_lower(a);
    let mut memory = dyn_stack::MemBuffer::new(
        faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(
            n,
            Par::Seq,
            faer::Spec::default(),
        ),
    );
    faer::linalg::cholesky::llt::factor::cholesky_in_place(
        lower.as_mut(),
        Default::default(),
        Par::Seq,
        dyn_stack::MemStack::new(&mut memory),
        faer::Spec::default(),
    )
    .ok()?;
    for col in 1..n {
        for row in 0..col {
            lower[(row, col)] = 0.0;
        }
    }
    Some(lower)
}

/// Solve `L Lᵀ x = rhs` for a factor from [`llt_factor_seq`], with `Par::Seq`.
fn llt_solve_seq(lower: MatRef<'_, f64>, rhs: MatRef<'_, f64>) -> Mat<f64> {
    let mut solution = rhs.to_owned();
    let stack = dyn_stack::MemStack::new(&mut []);
    faer::linalg::cholesky::llt::solve::solve_in_place_with_conj(
        lower,
        faer::Conj::No,
        solution.as_mut(),
        Par::Seq,
        stack,
    );
    solution
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
    if serial_inner() {
        return solve_linear_system_serial(a_ref, b_ref);
    }
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

/// [`solve_linear_system_ref`] with every factorisation and solve on one thread.
fn solve_linear_system_serial(
    a_ref: MatRef<'_, f64>,
    b_ref: MatRef<'_, f64>,
) -> Result<Mat<f64>, TwoPartError> {
    if let Some(lower) = llt_factor_seq(a_ref) {
        let solution = llt_solve_seq(lower.as_ref(), b_ref);
        if matrix_is_finite(&solution) {
            return Ok(solution);
        }
    }
    // The pivoted fallbacks are rare and p-by-p. They run on a one-worker pool, so
    // faer's global parallelism resolves to one thread there.
    let (a, b) = (a_ref.to_owned(), b_ref.to_owned());
    on_single_thread(|| {
        let lblt_solution = a.as_ref().lblt(Side::Lower).solve(b.as_ref());
        if matrix_is_finite(&lblt_solution) {
            return Ok(lblt_solution);
        }
        let partial_solution = a.as_ref().partial_piv_lu().solve(b.as_ref());
        if matrix_is_finite(&partial_solution) {
            return Ok(partial_solution);
        }
        let robust_solution = a.as_ref().full_piv_lu().solve(b.as_ref());
        if matrix_is_finite(&robust_solution) {
            Ok(robust_solution)
        } else {
            Err(TwoPartError::SolveFailed)
        }
    })
}

/// # Errors
///
/// Returns `TwoPartError::SolveFailed` if the solve produces non-finite values.
pub fn solve_linear_system(a: &Mat<f64>, b: &Mat<f64>) -> Result<Mat<f64>, TwoPartError> {
    solve_linear_system_ref(a.as_ref(), b.as_ref())
}

/// A factorization of a fixed left-hand side, cached so an IRLS loop whose
/// information matrix `X'WX` is invariant across iterations factors it once and
/// only back-substitutes the changing right-hand side each pass.
///
/// This applies to log-link GLMs whose Fisher-scoring weight `μ^(2-power)` is
/// β-independent — the gamma / Tweedie-power-2 case, where `W` is the fixed
/// sample weights — so the cached factor is the *exact* information matrix at
/// every iterate. The build mirrors [`solve_linear_system_ref`]: a Cholesky of
/// the symmetric, positive-definite ridged Gram matrix, with incremental jitter
/// on the rare failure, falling back to a pivoted LU.
pub(crate) enum CachedFactor {
    Llt(Llt<f64>),
    /// The lower Cholesky factor, built and applied with `Par::Seq`.
    LltSeq(Mat<f64>),
    Lu(PartialPivLu<f64>),
}

impl CachedFactor {
    /// Factor a (ridged) Gram matrix `X'WX` once for reuse across IRLS iterations.
    pub(crate) fn factor(matrix: &Mat<f64>) -> Self {
        if serial_inner() {
            return Self::factor_serial(matrix);
        }
        if let Ok(llt) = matrix.as_ref().llt(Side::Lower) {
            return Self::Llt(llt);
        }
        // Cholesky failed (numerically indefinite); jitter the diagonal and retry,
        // matching the stabilization the per-iteration solver applies on failure.
        let dim = matrix.nrows().min(matrix.ncols());
        let mut jitter = 1e-10;
        for _ in 0..8 {
            let stabilized = Mat::from_fn(matrix.nrows(), matrix.ncols(), |r, c| {
                if r == c && r < dim {
                    matrix[(r, c)] + jitter
                } else {
                    matrix[(r, c)]
                }
            });
            if let Ok(llt) = stabilized.as_ref().llt(Side::Lower) {
                return Self::Llt(llt);
            }
            jitter *= 10.0;
        }
        Self::Lu(matrix.as_ref().partial_piv_lu())
    }

    /// [`CachedFactor::factor`] with every factorisation on one thread.
    fn factor_serial(matrix: &Mat<f64>) -> Self {
        if let Some(lower) = llt_factor_seq(matrix.as_ref()) {
            return Self::LltSeq(lower);
        }
        let dim = matrix.nrows().min(matrix.ncols());
        let mut jitter = 1e-10;
        for _ in 0..8 {
            let stabilized = Mat::from_fn(matrix.nrows(), matrix.ncols(), |r, c| {
                if r == c && r < dim {
                    matrix[(r, c)] + jitter
                } else {
                    matrix[(r, c)]
                }
            });
            if let Some(lower) = llt_factor_seq(stabilized.as_ref()) {
                return Self::LltSeq(lower);
            }
            jitter *= 10.0;
        }
        let owned = matrix.clone();
        Self::Lu(on_single_thread(|| owned.as_ref().partial_piv_lu()))
    }

    /// Back-substitute the cached factor against `rhs`. Returns `SolveFailed` if the
    /// solution is non-finite (a pathological RHS), so the caller can fall back.
    pub(crate) fn solve(&self, rhs: MatRef<'_, f64>) -> Result<Mat<f64>, TwoPartError> {
        let solution = match self {
            Self::Llt(factor) => factor.solve(rhs),
            Self::LltSeq(lower) => llt_solve_seq(lower.as_ref(), rhs),
            Self::Lu(factor) if serial_inner() => {
                let rhs = rhs.to_owned();
                on_single_thread(|| factor.solve(rhs.as_ref()))
            }
            Self::Lu(factor) => factor.solve(rhs),
        };
        if matrix_is_finite(&solution) {
            Ok(solution)
        } else {
            Err(TwoPartError::SolveFailed)
        }
    }
}

/// IRLS weights for a log-link GLM whose Fisher-scoring weight is β-independent
/// (gamma / Tweedie power 2): the working weight is exactly the fixed prior
/// (sample/bootstrap) weight, so it is constant across iterations.
///
/// Reproduces the per-iteration computation bit-for-bit. The Tweedie loop forms
/// `base_weight * (μ^(2-power)).max(min_weight)`; at power 2 the variance factor
/// is `1.0` (above `min_weight`), so the weight is precisely `base_weight` with no
/// flooring of the prior weight itself — matched here.
#[must_use]
pub(crate) fn constant_irls_weights(
    sample_weights: Option<MatRef<'_, f64>>,
    n_rows: usize,
) -> Mat<f64> {
    Mat::from_fn(n_rows, 1, |i, _| sample_weights.map_or(1.0, |w| w[(i, 0)]))
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

/// Rows per block in every sum over the rows of a design matrix.
///
/// Each block is reduced on one thread with faer's sequential kernels, and the block
/// partials are added in block order. The partition depends only on the number of rows,
/// so the result is bit-identical for any thread count and any scheduling. faer's own
/// parallel kernels are not used for these products: its parallel matrix-vector product
/// splits the inner dimension across threads and adds the pieces, so its result depends
/// on the thread count.
const ROW_BLOCK: usize = 4096;

/// Sum of the `rows × cols` partials `block_term` writes for each block of `n` rows,
/// added in block order. Blocks are evaluated in parallel.
fn sum_over_row_blocks<F>(n: usize, rows: usize, cols: usize, block_term: F) -> Mat<f64>
where
    F: Fn(Range<usize>, MatMut<'_, f64>) + Sync,
{
    let mut total = Mat::<f64>::zeros(rows, cols);
    if n <= ROW_BLOCK {
        block_term(0..n, total.as_mut());
        return total;
    }
    let blocks = n.div_ceil(ROW_BLOCK);
    if serial_inner() {
        // The same partials added in the same block order, on this thread.
        for block in 0..blocks {
            let mut partial = Mat::<f64>::zeros(rows, cols);
            block_term(
                block * ROW_BLOCK..((block + 1) * ROW_BLOCK).min(n),
                partial.as_mut(),
            );
            add_in_place(&mut total, &partial);
        }
        return total;
    }
    let partials: Vec<Mat<f64>> = (0..blocks)
        .into_par_iter()
        .map(|block| {
            let mut partial = Mat::<f64>::zeros(rows, cols);
            block_term(
                block * ROW_BLOCK..((block + 1) * ROW_BLOCK).min(n),
                partial.as_mut(),
            );
            partial
        })
        .collect();
    for partial in &partials {
        add_in_place(&mut total, partial);
    }
    total
}

fn add_in_place(total: &mut Mat<f64>, partial: &Mat<f64>) {
    for col in 0..total.ncols() {
        for row in 0..total.nrows() {
            total[(row, col)] += partial[(row, col)];
        }
    }
}

/// `Σ_i s_i² x_i x_iᵀ` for the row scales `s_i = scale(i)`, exactly symmetric.
///
/// Each block forms `S X` for its rows and adds `(S X)ᵀ (S X)` through a symmetric
/// rank-k update of the lower triangle, which is then mirrored.
pub(crate) fn row_scaled_gram(x: MatRef<'_, f64>, scale: impl Fn(usize) -> f64 + Sync) -> Mat<f64> {
    let p = x.ncols();
    let mut gram = sum_over_row_blocks(x.nrows(), p, p, |rows, out| {
        let start = rows.start;
        // One scale per row rather than one per element: the same values, p times fewer calls.
        let scales: Vec<f64> = rows.clone().map(&scale).collect();
        let scaled = Mat::from_fn(rows.len(), p, |row, col| {
            x[(start + row, col)] * scales[row]
        });
        triangular::matmul(
            out,
            BlockStructure::TriangularLower,
            Accum::Replace,
            scaled.transpose(),
            BlockStructure::Rectangular,
            scaled.as_ref(),
            BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );
    });
    for col in 0..p {
        for row in (col + 1)..p {
            gram[(col, row)] = gram[(row, col)];
        }
    }
    gram
}

/// `Xᵀ v` for a column `v` of the same length as `X`.
pub(crate) fn xt_column(x: MatRef<'_, f64>, v: &[f64]) -> Mat<f64> {
    sum_over_row_blocks(x.nrows(), x.ncols(), 1, |rows, out| {
        let start = rows.start;
        let len = rows.len();
        let block = MatRef::from_column_major_slice(&v[rows], len, 1);
        matmul(
            out,
            Accum::Replace,
            x.as_ref().subrows(start, len).transpose(),
            block,
            1.0,
            Par::Seq,
        );
    })
}

/// `X'WX` with the weights floored at zero, exactly symmetric.
#[must_use]
pub fn weighted_xtx(x: &Mat<f64>, weights: &Mat<f64>) -> Mat<f64> {
    weighted_xtx_ref(x.as_ref(), weights.as_ref())
}

/// [`weighted_xtx`] on borrowed matrices.
#[must_use]
pub fn weighted_xtx_ref(x: MatRef<'_, f64>, weights: MatRef<'_, f64>) -> Mat<f64> {
    row_scaled_gram(x, |row| weights[(row, 0)].max(0.0).sqrt())
}

/// In-place matrix-vector product `dst <- x * beta`, reusing `dst`'s storage so an
/// IRLS loop does not allocate a fresh linear-predictor column every iteration.
///
/// Rows are split into fixed blocks, so each element is the same sequential sum over
/// the columns whatever the thread count.
pub(crate) fn matvec_into(dst: &mut Mat<f64>, x: MatRef<'_, f64>, beta: &Mat<f64>) {
    let n = x.nrows();
    if n <= ROW_BLOCK {
        matmul(
            dst.as_mut(),
            Accum::Replace,
            x.as_ref(),
            beta.as_ref(),
            1.0,
            Par::Seq,
        );
        return;
    }
    let column = dst.col_as_slice_mut(0);
    if serial_inner() {
        for (block, chunk) in column.chunks_mut(ROW_BLOCK).enumerate() {
            matvec_block(chunk, x, beta, block);
        }
    } else {
        column
            .par_chunks_mut(ROW_BLOCK)
            .enumerate()
            .for_each(|(block, chunk)| matvec_block(chunk, x, beta, block));
    }
}

/// `chunk <- x[block rows] * beta` for the rows of block number `block`.
fn matvec_block(chunk: &mut [f64], x: MatRef<'_, f64>, beta: &Mat<f64>, block: usize) {
    let len = chunk.len();
    matmul(
        MatMut::from_column_major_slice_mut(chunk, len, 1),
        Accum::Replace,
        x.subrows(block * ROW_BLOCK, len),
        beta.as_ref(),
        1.0,
        Par::Seq,
    );
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
    weighted_xtz_ref_with_buffer(
        x.as_ref(),
        weights.as_ref(),
        response.as_ref(),
        weighted_buffer,
    )
}

/// [`weighted_xtz_with_buffer`] on borrowed matrices.
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
    for row in 0..n {
        weighted_buffer[row] = weights[(row, 0)] * response[(row, 0)];
    }
    xt_column(x, &weighted_buffer[..n])
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
        return EffectIntervalSummary::default();
    }

    let mean = values.iter().sum::<f64>() / usize_to_f64(values.len());
    values.sort_by(f64::total_cmp);

    EffectIntervalSummary {
        mean,
        q025: calculate_quantile(values, 0.025),
        q50: calculate_quantile(values, 0.5),
        q975: calculate_quantile(values, 0.975),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Mutex;

    #[test]
    fn serial_scope_keeps_row_blocks_on_the_calling_thread() {
        let n = 5 * ROW_BLOCK + 17;
        let x = Mat::from_fn(n, 3, |i, j| ((i * 7 + j * 3) % 11) as f64);
        let threads_used = |serial: bool| {
            let seen = Mutex::new(HashSet::new());
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .expect("pool");
            let gram = pool.install(|| {
                let _scope = SerialInnerScope::enter(serial);
                row_scaled_gram(x.as_ref(), |row| {
                    seen.lock()
                        .expect("lock")
                        .insert(std::thread::current().id());
                    1.0 + (row % 3) as f64
                })
            });
            (gram, seen.into_inner().expect("lock").len())
        };
        let (serial_gram, serial_threads) = threads_used(true);
        let (parallel_gram, _) = threads_used(false);
        assert_eq!(serial_threads, 1);
        assert_eq!(serial_gram, parallel_gram);
    }
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
