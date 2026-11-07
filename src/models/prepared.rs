//! Repeated gamma and two-part fits of one design under different weights.
//!
//! A fractional-random-weight bootstrap fits the same rows thousands of times with only
//! the weights changing. The types here read the design once into a compact form: for
//! each row the columns that hold a 1, and the values of the few columns that are not
//! binary. Every product then runs on that form, and the centring the dense fits apply
//! to the design is applied by algebra instead:
//!
//! ```text
//! X~'WX~ = X'WX - W m m'     X~'Wz = X'Wz - m (1'Wz)
//! ```
//!
//! where `m` holds the weighted column means and `W` the sum of the weights. Non-binary
//! columns are stored with their unweighted mean subtracted, which removes most of the
//! cancellation from that subtraction and leaves every slope unchanged.
//!
//! # Agreement with the dense fits
//!
//! The fits take the steps of [`fit_tweedie_ref_warm_prevalidated`] at power 2 and of
//! [`fit_two_part_ref_warm_prevalidated`]: the same start, step, line search and stopping
//! rule. The arithmetic is done in another order, so results agree up to rounding and are
//! not bit-identical to the dense fits. A fit here is deterministic in its inputs and does
//! not depend on the thread it runs on or on what its scratch held before.
//!
//! # When to fall back to the dense fits
//!
//! The types are built for a caller that holds a dense fit as well, and uses these while
//! they apply:
//!
//! - A constructor returns `None` for a design the compact form cannot represent: column
//!   0 is not a constant 1, another column is constant among the rows being fitted, there
//!   are more than [`MAX_COLUMNS`] columns, or a gamma outcome is not positive. The caller
//!   uses the dense fit for every replicate.
//! - A fit returns a [`PreparedFitError`] when it cannot give a reliable answer for these
//!   weights. It makes one attempt, without the retries of [`FitStrategy::Relaxed`], and
//!   refuses an information matrix that is close to singular, where the outcome would
//!   depend on rounding. The caller runs the dense fit for that replicate, which applies
//!   its own stabilisation and retries.
//!
//! The rows should be many. On a small sample with rare levels the information matrix of
//! a replicate is often nearly singular, and the two paths are then furthest apart.
//!
//! # Example
//!
//! ```
//! use faer::Mat;
//! use semicontinuous_models::{
//!     ModelInput, PreparedGammaDesign, PreparedScratch, TweedieOptions,
//!     fit_tweedie_ref_warm_prevalidated,
//! };
//!
//! let n = 2_000;
//! let x = Mat::from_fn(n, 3, |i, j| match j {
//!     0 => 1.0,
//!     1 => f64::from(i % 3 == 0),
//!     _ => (i % 50) as f64,
//! });
//! let y = Mat::from_fn(n, 1, |i, _| 100.0 + 40.0 * x[(i, 1)] + (i % 17) as f64);
//! let input = ModelInput::new(x.clone(), y.clone());
//! let options = TweedieOptions::default();
//! let (point, _) = fit_tweedie_ref_warm_prevalidated(input.as_ref(), 2.0, options, None).unwrap();
//!
//! let design = PreparedGammaDesign::new(x.as_ref(), y.col_as_slice(0)).expect("representable");
//! let mut scratch = PreparedScratch::default();
//! let weights: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64 / 7.0).collect();
//! let beta = match design.fit(&weights, &point.beta, options, &mut scratch) {
//!     Ok(fit) => fit.beta,
//!     Err(_) => {
//!         let w = Mat::from_fn(n, 1, |i, _| weights[i]);
//!         let input = ModelInput::new(x.clone(), y.clone()).with_sample_weights(w);
//!         fit_tweedie_ref_warm_prevalidated(input.as_ref(), 2.0, options, Some(&point.beta))
//!             .unwrap()
//!             .0
//!             .beta
//!     }
//! };
//! assert_eq!(beta.nrows(), 3);
//! ```
//!
//! [`fit_tweedie_ref_warm_prevalidated`]: crate::fit_tweedie_ref_warm_prevalidated
//! [`fit_two_part_ref_warm_prevalidated`]: crate::fit_two_part_ref_warm_prevalidated
//! [`FitStrategy::Relaxed`]: crate::models::FitStrategy::Relaxed

use faer::{Mat, MatRef};

use crate::models::matrix_ops::column_is_constant;
use crate::models::tweedie::{
    BACKTRACKING_STEPS, LINEAR_PREDICTOR_CLIP, TweedieOptions, exp_clamped, gamma_deviance,
    relative_change,
};
use crate::models::two_part::irls::{
    GAMMA_LOG_LINK_ETA_CLAMP, LOGIT_ETA_CLAMP, ridge_from_regularization,
};
use crate::models::two_part::{FitOptions, Regularization, TwoPartWarmStart};
use crate::utils::{add_ridge_to_diagonal, llt_factor_seq, llt_solve_seq, matrix_is_finite};

/// The most columns a compact design holds: a row's list stores column numbers in a byte.
pub const MAX_COLUMNS: usize = 255;

/// Smallest accepted share of a Cholesky pivot in its diagonal element.
///
/// The pivot of column `j` is the part of the column not explained by the columns before
/// it, so the share is `1 - R^2` of that regression. Exactly collinear columns leave a
/// share near machine epsilon, and the fit is then decided by rounding in the matrix. The
/// spline terms of an age column legitimately reach about 1e-7, well above this bound.
const MIN_PIVOT_SHARE: f64 = 1e-10;

/// Smallest accepted ratio of a centred diagonal element to the raw one.
///
/// A column that is nearly constant under a replicate's weights loses about `-log10` of
/// this ratio in digits when the centring is subtracted.
const MIN_CENTRED_SHARE: f64 = 1e-6;

/// Why a compact fit gave no result. The caller runs the dense fit for that replicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PreparedFitError {
    /// The weights are not one finite, strictly positive value per row.
    #[error("weights must be one finite, strictly positive value per row")]
    InvalidWeights,
    /// The warm start is not one column with one coefficient per design column.
    #[error("the warm start must have one coefficient per design column")]
    InvalidWarmStart,
    /// The options ask for a penalty the compact fit does not apply.
    #[error("this regularization is not available on a compact design")]
    UnsupportedRegularization,
    /// The information matrix is too close to singular for a reliable compact fit.
    #[error("the information matrix is too close to singular")]
    IllConditioned,
    /// A solve or an iterate gave non-finite values.
    #[error("the fit gave non-finite values")]
    NonFinite,
    /// The converged linear predictor reached the log-link bound.
    #[error("the linear predictor reached the log-link bound")]
    Saturated,
    /// The iteration did not converge within the allowed iterations.
    #[error("the fit did not converge")]
    NonConvergence,
}

/// The rows of a design: a list of the 1s per row, and the other columns in row order.
#[derive(Debug, Clone)]
struct CompactRows {
    n: usize,
    p: usize,
    binary_cols: Vec<usize>,
    dense_cols: Vec<usize>,
    /// What was subtracted from each design column: 0 for column 0 and for binary columns.
    shifts: Vec<f64>,
    /// Row-major values of the non-binary columns, after the shift.
    dense: Vec<f64>,
    /// Row `i` holds a 1 in the columns `ones[offsets[i]..offsets[i + 1]]`, ascending.
    offsets: Vec<u32>,
    ones: Vec<u8>,
}

/// The cross products of a design under one weight column.
#[derive(Debug, Default)]
struct Products {
    /// The lower triangle of `X'WX` as a `p` by `p` table. Row and column 0 stay unused.
    lower: Vec<f64>,
    /// `X'w`, with the sum of the weights in position 0.
    sums: Vec<f64>,
    dense_dense: Vec<f64>,
    binary_dense: Vec<f64>,
    weighted_row: Vec<f64>,
    dense_sums: Vec<f64>,
}

impl CompactRows {
    #[allow(clippy::float_cmp, clippy::cast_precision_loss)]
    fn new(x: MatRef<'_, f64>) -> Option<Self> {
        let (n, p) = (x.nrows(), x.ncols());
        if n == 0 || p == 0 || p > MAX_COLUMNS {
            return None;
        }
        let columns: Vec<&[f64]> = (0..p)
            .map(|j| x.col(j).try_as_col_major().map(|column| column.as_slice()))
            .collect::<Option<_>>()?;
        if columns[0].iter().any(|&value| value != 1.0) {
            return None;
        }
        let (binary_cols, dense_cols): (Vec<usize>, Vec<usize>) =
            (1..p).partition(|&j| columns[j].iter().all(|&value| value == 0.0 || value == 1.0));

        let d = dense_cols.len();
        let mut shifts = vec![0.0_f64; p];
        let mut dense = vec![0.0_f64; n * d];
        for (k, &j) in dense_cols.iter().enumerate() {
            let shift = columns[j].iter().sum::<f64>() / n as f64;
            shifts[j] = shift;
            for (i, &value) in columns[j].iter().enumerate() {
                dense[i * d + k] = value - shift;
            }
        }

        // Count the 1s of each row, then fill column by column so each list ascends.
        let mut offsets = vec![0_u32; n + 1];
        for &j in &binary_cols {
            for (i, &value) in columns[j].iter().enumerate() {
                if value != 0.0 {
                    offsets[i + 1] += 1;
                }
            }
        }
        for i in 0..n {
            offsets[i + 1] = offsets[i + 1].checked_add(offsets[i])?;
        }
        let mut next = offsets.clone();
        let mut ones = vec![0_u8; offsets[n] as usize];
        for &j in &binary_cols {
            let column_number = u8::try_from(j).ok()?;
            for (i, &value) in columns[j].iter().enumerate() {
                if value != 0.0 {
                    ones[next[i] as usize] = column_number;
                    next[i] += 1;
                }
            }
        }

        Some(Self {
            n,
            p,
            binary_cols,
            dense_cols,
            shifts,
            dense,
            offsets,
            ones,
        })
    }

    #[inline]
    fn row_ones(&self, i: usize) -> &[u8] {
        &self.ones[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    /// The linear predictor of row `i` on the shifted design.
    #[inline]
    fn row_eta(&self, i: usize, intercept: f64, slopes: &[f64], dense_slopes: &[f64]) -> f64 {
        let mut acc = intercept;
        for &column in self.row_ones(i) {
            acc += slopes[column as usize];
        }
        let d = dense_slopes.len();
        if d > 0 {
            for (&value, &slope) in self.dense[i * d..(i + 1) * d].iter().zip(dense_slopes) {
                acc += value * slope;
            }
        }
        acc
    }

    fn select_dense(&self, slopes: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.extend(self.dense_cols.iter().map(|&j| slopes[j]));
    }

    /// `eta = intercept + X' slopes` on the shifted design. Position 0 of `slopes` is
    /// not read.
    fn linear_predictor(
        &self,
        intercept: f64,
        slopes: &[f64],
        dense_slopes: &mut Vec<f64>,
        eta: &mut [f64],
    ) {
        self.select_dense(slopes, dense_slopes);
        for (i, slot) in eta.iter_mut().enumerate() {
            *slot = self.row_eta(i, intercept, slopes, dense_slopes);
        }
    }

    /// The intercept on the shifted design of coefficients on the raw design.
    fn shifted_from_raw(&self, raw: &[f64]) -> f64 {
        raw[0] + (1..self.p).map(|j| raw[j] * self.shifts[j]).sum::<f64>()
    }

    /// The raw-design intercept of an intercept on the shifted design.
    fn raw_from_shifted(&self, shifted: f64, slopes: &[f64]) -> f64 {
        shifted - (1..self.p).map(|j| slopes[j] * self.shifts[j]).sum::<f64>()
    }

    /// `out = X'w` on the shifted design, with the sum of `w` in position 0.
    fn weighted_sums(&self, w: &[f64], dense_acc: &mut Vec<f64>, out: &mut [f64]) {
        let d = self.dense_cols.len();
        out.fill(0.0);
        dense_acc.clear();
        dense_acc.resize(d, 0.0);
        let mut total = 0.0;
        for (i, &wi) in w.iter().enumerate() {
            total += wi;
            for &column in self.row_ones(i) {
                out[column as usize] += wi;
            }
            if d > 0 {
                for (acc, &value) in dense_acc.iter_mut().zip(&self.dense[i * d..(i + 1) * d]) {
                    *acc += wi * value;
                }
            }
        }
        out[0] = total;
        for (&j, &value) in self.dense_cols.iter().zip(dense_acc.iter()) {
            out[j] = value;
        }
    }

    /// The lower triangle of `X'WX` and the sums `X'w` on the shifted design, from one
    /// pass over the rows.
    fn weighted_gram(&self, w: &[f64], products: &mut Products) {
        let (p, d) = (self.p, self.dense_cols.len());
        let Products {
            lower,
            sums,
            dense_dense,
            binary_dense,
            weighted_row,
            dense_sums,
        } = products;
        for (buffer, len) in [
            (&mut *lower, p * p),
            (&mut *sums, p),
            (&mut *dense_dense, d * d),
            (&mut *binary_dense, p * d),
            (&mut *weighted_row, d),
            (&mut *dense_sums, d),
        ] {
            buffer.clear();
            buffer.resize(len, 0.0);
        }
        let mut total = 0.0;
        for (i, &wi) in w.iter().enumerate() {
            total += wi;
            let row = self.row_ones(i);
            for (position, &a) in row.iter().enumerate() {
                let a = a as usize;
                sums[a] += wi;
                let line = &mut lower[a * p..a * p + p];
                for &b in &row[..=position] {
                    line[b as usize] += wi;
                }
            }
            if d > 0 {
                let values = &self.dense[i * d..(i + 1) * d];
                for ((slot, &value), acc) in weighted_row
                    .iter_mut()
                    .zip(values)
                    .zip(dense_sums.iter_mut())
                {
                    *slot = wi * value;
                    *acc += *slot;
                }
                for k in 0..d {
                    let line = &mut dense_dense[k * d..k * d + d];
                    let weighted = weighted_row[k];
                    for (cell, &value) in line[..=k].iter_mut().zip(values) {
                        *cell += weighted * value;
                    }
                }
                for &a in row {
                    let a = a as usize;
                    let line = &mut binary_dense[a * d..(a + 1) * d];
                    for (cell, &value) in line.iter_mut().zip(weighted_row.iter()) {
                        *cell += value;
                    }
                }
            }
        }
        sums[0] = total;
        for (k, &j) in self.dense_cols.iter().enumerate() {
            sums[j] = dense_sums[k];
            for (l, &m) in self.dense_cols.iter().enumerate().take(k + 1) {
                lower[j.max(m) * p + j.min(m)] = dense_dense[k * d + l];
            }
            for &a in &self.binary_cols {
                lower[a.max(j) * p + a.min(j)] = binary_dense[a * d + k];
            }
        }
    }
}

/// The information matrix of the centred design and the weighted column means of the
/// shifted design.
fn centred_information(
    products: &Products,
    p: usize,
    means: &mut Vec<f64>,
) -> Result<Mat<f64>, PreparedFitError> {
    let (lower, sums) = (&products.lower, &products.sums);
    let total = sums[0];
    means.clear();
    means.push(0.0);
    means.extend(sums[1..].iter().map(|&sum| sum / total));
    let information = Mat::from_fn(p, p, |row, col| {
        let (high, low) = (row.max(col), row.min(col));
        if high == 0 {
            total
        } else if low == 0 {
            0.0
        } else {
            lower[high * p + low] - total * means[high] * means[low]
        }
    });
    // Written so that a NaN is refused as well.
    if (1..p).any(|j| {
        let keeps_enough = information[(j, j)] >= MIN_CENTRED_SHARE * lower[j * p + j];
        !keeps_enough
    }) {
        return Err(PreparedFitError::IllConditioned);
    }
    Ok(information)
}

/// The information matrix on the shifted design with no centring: the logit fit's.
fn uncentred_information(products: &Products, p: usize) -> Mat<f64> {
    let (lower, sums) = (&products.lower, &products.sums);
    Mat::from_fn(p, p, |row, col| {
        let (high, low) = (row.max(col), row.min(col));
        if low == 0 {
            sums[high]
        } else {
            lower[high * p + low]
        }
    })
}

/// The Cholesky factor of `information`, on one thread, when every pivot keeps at least
/// [`MIN_PIVOT_SHARE`] of its diagonal element.
fn factor_checked(information: &Mat<f64>) -> Result<Mat<f64>, PreparedFitError> {
    let lower = llt_factor_seq(information.as_ref()).ok_or(PreparedFitError::IllConditioned)?;
    // Written so that a NaN is refused as well.
    if (0..information.nrows()).any(|j| {
        let pivot = lower[(j, j)];
        let keeps_enough = pivot * pivot >= MIN_PIVOT_SHARE * information[(j, j)];
        !keeps_enough
    }) {
        return Err(PreparedFitError::IllConditioned);
    }
    Ok(lower)
}

fn solve_checked(lower: &Mat<f64>, rhs: &Mat<f64>) -> Result<Mat<f64>, PreparedFitError> {
    let solution = llt_solve_seq(lower.as_ref(), rhs.as_ref());
    if matrix_is_finite(&solution) {
        Ok(solution)
    } else {
        Err(PreparedFitError::NonFinite)
    }
}

fn check_weights(weights: &[f64], n: usize) -> Result<(), PreparedFitError> {
    if weights.len() == n && weights.iter().all(|&w| w.is_finite() && w > 0.0) {
        Ok(())
    } else {
        Err(PreparedFitError::InvalidWeights)
    }
}

fn warm_values(warm: &Mat<f64>, p: usize, out: &mut Vec<f64>) -> Result<(), PreparedFitError> {
    if warm.nrows() != p || warm.ncols() != 1 {
        return Err(PreparedFitError::InvalidWarmStart);
    }
    out.clear();
    out.extend_from_slice(warm.col_as_slice(0));
    Ok(())
}

fn column_vector(values: &[f64]) -> Mat<f64> {
    Mat::from_fn(values.len(), 1, |j, _| values[j])
}

fn column_buffer(buffer: &mut Vec<f64>, len: usize) -> &mut [f64] {
    if buffer.len() < len {
        buffer.resize(len, 0.0);
    }
    &mut buffer[..len]
}

/// Working memory of the compact fits.
///
/// One scratch serves one fit at a time, so a caller that fits in parallel keeps one per
/// worker. It carries nothing from one fit to the next, and a result does not depend on
/// what it held before.
#[derive(Debug, Default)]
pub struct PreparedScratch {
    products: Products,
    means: Vec<f64>,
    dense_slopes: Vec<f64>,
    dense_acc: Vec<f64>,
    cross: Vec<f64>,
    beta: Vec<f64>,
    candidate: Vec<f64>,
    proposal: Vec<f64>,
    best: Vec<f64>,
    eta: Vec<f64>,
    mu: Vec<f64>,
    search_eta: Vec<f64>,
    search_mu: Vec<f64>,
    weighted_response: Vec<f64>,
    irls_weights: Vec<f64>,
    positive_weights: Vec<f64>,
}

/// The result of [`PreparedGammaDesign::fit`].
#[derive(Debug, Clone)]
pub struct PreparedGammaFit {
    /// Coefficients on the raw design, as the dense fit returns them.
    pub beta: Mat<f64>,
    /// Iterations taken.
    pub iterations: usize,
}

/// The result of [`PreparedTwoPartDesign::fit`].
#[derive(Debug, Clone)]
pub struct PreparedTwoPartFit {
    /// Logit coefficients on the raw design.
    pub beta_logit: Mat<f64>,
    /// Gamma coefficients on the raw design.
    pub beta_gamma: Mat<f64>,
    /// Iterations of the logit part, 0 when it is an intercept-only prevalence model.
    pub iterations_logit: usize,
    /// Iterations of the gamma part.
    pub iterations_gamma: usize,
}

/// A design and a positive outcome, for gamma log-link fits under many weight columns.
#[derive(Debug, Clone)]
pub struct PreparedGammaDesign {
    rows: CompactRows,
    y: Vec<f64>,
}

impl PreparedGammaDesign {
    /// Read `x` and `y`, or `None` when the compact fit does not apply: see the
    /// [module documentation](self#when-to-fall-back-to-the-dense-fits). The design is
    /// also refused when its columns are not contiguous in memory.
    #[must_use]
    pub fn new(x: MatRef<'_, f64>, y: &[f64]) -> Option<Self> {
        if y.len() != x.nrows() || y.iter().any(|&value| value.is_nan() || value <= 0.0) {
            return None;
        }
        if (1..x.ncols()).any(|j| column_is_constant(x, j)) {
            return None;
        }
        Some(Self {
            rows: CompactRows::new(x)?,
            y: y.to_vec(),
        })
    }

    /// Rows of the design.
    #[must_use]
    pub const fn nrows(&self) -> usize {
        self.rows.n
    }

    /// Columns of the design.
    #[must_use]
    pub const fn ncols(&self) -> usize {
        self.rows.p
    }

    /// The gamma log-link fit under `weights`, from the raw-scale coefficients `warm`.
    ///
    /// It takes the steps of `fit_tweedie_ref_warm_prevalidated` at power 2, reading
    /// `max_iter`, `tolerance`, `l2_penalty` and `l2_penalty_exclude_intercept` from
    /// `options`. It makes one attempt and computes no standard errors.
    ///
    /// # Errors
    ///
    /// A [`PreparedFitError`] when these weights give no reliable compact fit. The caller
    /// runs the dense fit for them.
    #[allow(clippy::too_many_lines)]
    pub fn fit(
        &self,
        weights: &[f64],
        warm: &Mat<f64>,
        options: TweedieOptions,
        scratch: &mut PreparedScratch,
    ) -> Result<PreparedGammaFit, PreparedFitError> {
        let rows = &self.rows;
        let (n, p) = (rows.n, rows.p);
        check_weights(weights, n)?;
        let PreparedScratch {
            products,
            means,
            dense_slopes,
            dense_acc,
            cross,
            beta,
            candidate,
            proposal,
            best,
            eta,
            mu,
            search_eta,
            search_mu,
            weighted_response,
            ..
        } = scratch;
        warm_values(warm, p, beta)?;

        rows.weighted_gram(weights, products);
        let mut information = centred_information(products, p, means)?;
        if options.l2_penalty > 0.0 {
            add_ridge_to_diagonal(
                &mut information,
                options.l2_penalty,
                options.l2_penalty_exclude_intercept,
            );
        }
        let factor = factor_checked(&information)?;

        // On the centred scale, as the dense fit iterates: `beta[0]` is the intercept of
        // the centred design and `eta = beta[0] + sum_j beta[j] (x'_j - m_j)` for the
        // shifted columns `x'_j` and their weighted means `m_j`.
        beta[0] = rows.shifted_from_raw(beta) + (1..p).map(|j| beta[j] * means[j]).sum::<f64>();
        let shifted_intercept = |centred: f64, slopes: &[f64]| {
            centred - (1..p).map(|j| slopes[j] * means[j]).sum::<f64>()
        };

        let wz = column_buffer(weighted_response, n);
        let mut eta_now = column_buffer(eta, n);
        let mut mu_now = column_buffer(mu, n);
        let mut eta_next = column_buffer(search_eta, n);
        let mut mu_next = column_buffer(search_mu, n);
        let cross = column_buffer(cross, p);
        let candidate = column_buffer(candidate, p);
        let proposal = column_buffer(proposal, p);
        let best = column_buffer(best, p);

        rows.linear_predictor(
            shifted_intercept(beta[0], beta),
            beta,
            dense_slopes,
            eta_now,
        );
        for (m, &e) in mu_now.iter_mut().zip(eta_now.iter()) {
            *m = exp_clamped(e);
        }
        let mut current_deviance = gamma_deviance(&self.y, mu_now, weights);

        for iteration in 0..options.max_iter {
            for (((slot, &w), (&e, &m)), &yi) in wz
                .iter_mut()
                .zip(weights)
                .zip(eta_now.iter().zip(mu_now.iter()))
                .zip(&self.y)
            {
                *slot = w * (e + (yi - m) / m);
            }
            rows.weighted_sums(wz, dense_acc, cross);
            let total = cross[0];
            let rhs = Mat::from_fn(p, 1, |j, _| {
                if j == 0 {
                    total
                } else {
                    cross[j] - means[j] * total
                }
            });
            let solved = solve_checked(&factor, &rhs)?;
            candidate.copy_from_slice(solved.col_as_slice(0));

            // The line search of the dense fit: halve the step until the deviance does
            // not rise, and otherwise take the best finite trial.
            let mut step = 1.0_f64;
            let mut best_deviance = f64::INFINITY;
            best.copy_from_slice(candidate);
            let mut accepted = None;
            for _ in 0..BACKTRACKING_STEPS {
                if (step - 1.0).abs() < f64::EPSILON {
                    proposal.copy_from_slice(candidate);
                } else {
                    for ((slot, &c), &b) in
                        proposal.iter_mut().zip(candidate.iter()).zip(beta.iter())
                    {
                        *slot = (c - b).mul_add(step, b);
                    }
                }
                rows.linear_predictor(
                    shifted_intercept(proposal[0], proposal),
                    proposal,
                    dense_slopes,
                    eta_next,
                );
                for (m, &e) in mu_next.iter_mut().zip(eta_next.iter()) {
                    *m = exp_clamped(e);
                }
                let trial = gamma_deviance(&self.y, mu_next, weights);
                if trial.is_finite() && trial <= current_deviance {
                    accepted = Some(trial);
                    break;
                }
                if trial.is_finite() && trial < best_deviance {
                    best_deviance = trial;
                    best.copy_from_slice(proposal);
                }
                step *= 0.5;
            }
            let next_deviance = accepted.unwrap_or_else(|| {
                proposal.copy_from_slice(best);
                rows.linear_predictor(
                    shifted_intercept(proposal[0], proposal),
                    proposal,
                    dense_slopes,
                    eta_next,
                );
                for (m, &e) in mu_next.iter_mut().zip(eta_next.iter()) {
                    *m = exp_clamped(e);
                }
                gamma_deviance(&self.y, mu_next, weights)
            });

            let beta_change = proposal
                .iter()
                .zip(beta.iter())
                .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
            if beta_change < options.tolerance
                || relative_change(current_deviance, next_deviance) < options.tolerance
            {
                let max_eta = eta_next.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
                if max_eta >= LINEAR_PREDICTOR_CLIP {
                    return Err(PreparedFitError::Saturated);
                }
                if proposal.iter().any(|value| !value.is_finite()) {
                    return Err(PreparedFitError::NonFinite);
                }
                proposal[0] =
                    rows.raw_from_shifted(shifted_intercept(proposal[0], proposal), proposal);
                return Ok(PreparedGammaFit {
                    beta: column_vector(proposal),
                    iterations: iteration + 1,
                });
            }
            std::mem::swap(&mut eta_now, &mut eta_next);
            std::mem::swap(&mut mu_now, &mut mu_next);
            current_deviance = next_deviance;
            beta.copy_from_slice(proposal);
        }
        Err(PreparedFitError::NonConvergence)
    }
}

/// A design and an outcome, for two-part fits under many weight columns: a logit on every
/// row and a gamma log-link on the rows with a positive outcome.
#[derive(Debug, Clone)]
pub struct PreparedTwoPartDesign {
    all: CompactRows,
    positive: CompactRows,
    positive_rows: Vec<usize>,
    is_positive: Vec<f64>,
    y_positive: Vec<f64>,
}

impl PreparedTwoPartDesign {
    /// Read `x` and `y`, or `None` when the compact fit does not apply: see the
    /// [module documentation](self#when-to-fall-back-to-the-dense-fits). No row with a
    /// positive outcome, and a column that is constant among those rows, also refuse the
    /// design.
    #[must_use]
    pub fn new(x: MatRef<'_, f64>, y: &[f64]) -> Option<Self> {
        if y.len() != x.nrows() {
            return None;
        }
        let positive_rows: Vec<usize> = (0..y.len()).filter(|&i| y[i] > 0.0).collect();
        if positive_rows.is_empty() {
            return None;
        }
        let gathered = crate::models::matrix_ops::select_rows_ref(x, &positive_rows);
        if (1..x.ncols()).any(|j| column_is_constant(gathered.as_ref(), j)) {
            return None;
        }
        Some(Self {
            all: CompactRows::new(x)?,
            positive: CompactRows::new(gathered.as_ref())?,
            is_positive: y.iter().map(|&value| f64::from(value > 0.0)).collect(),
            y_positive: positive_rows.iter().map(|&i| y[i]).collect(),
            positive_rows,
        })
    }

    /// Rows of the design.
    #[must_use]
    pub const fn nrows(&self) -> usize {
        self.all.n
    }

    /// Rows with a positive outcome, which the gamma part is fitted on.
    #[must_use]
    pub const fn positive_nrows(&self) -> usize {
        self.positive.n
    }

    /// Columns of the design.
    #[must_use]
    pub const fn ncols(&self) -> usize {
        self.all.p
    }

    /// The logit and gamma fits under `weights`, from the raw-scale coefficients `warm`.
    ///
    /// They take the steps of `fit_two_part_ref_warm_prevalidated`, reading `max_iter`,
    /// `tolerance`, `deviance_tolerance`, `max_step_halvings`, `min_weight` and
    /// `regularization` from `options`. It makes one attempt and computes no covariance.
    ///
    /// # Errors
    ///
    /// A [`PreparedFitError`] when these weights give no reliable compact fit, including
    /// an elastic net or a ridge on the intercept, which are not available here. The
    /// caller runs the dense fit for them.
    pub fn fit(
        &self,
        weights: &[f64],
        warm: &TwoPartWarmStart,
        options: FitOptions,
        scratch: &mut PreparedScratch,
    ) -> Result<PreparedTwoPartFit, PreparedFitError> {
        let ridge = match options.regularization {
            Regularization::ElasticNet { .. } => {
                return Err(PreparedFitError::UnsupportedRegularization);
            }
            other => ridge_from_regularization(other),
        };
        // A ridge on the intercept would not carry over to the shifted design.
        if matches!(ridge, Some((lambda, false)) if lambda > 0.0) {
            return Err(PreparedFitError::UnsupportedRegularization);
        }
        let ridge = ridge.map_or(0.0, |(lambda, _)| lambda);

        check_weights(weights, self.all.n)?;
        let (beta_logit, iterations_logit) =
            self.fit_logit(weights, &warm.beta_logit, ridge, options, scratch)?;
        let (beta_gamma, iterations_gamma) =
            self.fit_gamma_part(weights, &warm.beta_gamma, ridge, options, scratch)?;
        Ok(PreparedTwoPartFit {
            beta_logit,
            beta_gamma,
            iterations_logit,
            iterations_gamma,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn fit_logit(
        &self,
        weights: &[f64],
        warm: &Mat<f64>,
        ridge: f64,
        options: FitOptions,
        scratch: &mut PreparedScratch,
    ) -> Result<(Mat<f64>, usize), PreparedFitError> {
        let design = &self.all;
        let (n, p) = (design.n, design.p);
        let y = &self.is_positive;

        // The minority class decides whether a slope is identified at all, as in the
        // dense fit: without enough of it the model is the intercept-only prevalence.
        let count_positive = self.positive.n;
        if count_positive.min(n - count_positive) <= p {
            let mut weighted_sum = 0.0;
            let mut weighted_positive = 0.0_f64;
            for (&yi, &wi) in y.iter().zip(weights) {
                weighted_sum += wi;
                weighted_positive = yi.mul_add(wi, weighted_positive);
            }
            let mut beta = Mat::<f64>::zeros(p, 1);
            let share = (weighted_positive / weighted_sum)
                .clamp(options.min_weight, 1.0 - options.min_weight);
            beta[(0, 0)] = (share / (1.0 - share)).ln();
            return Ok((beta, 0));
        }

        let PreparedScratch {
            products,
            dense_slopes,
            dense_acc,
            cross,
            beta,
            candidate,
            eta,
            mu,
            search_eta,
            search_mu,
            weighted_response,
            irls_weights,
            ..
        } = scratch;
        warm_values(warm, p, beta)?;
        // On the shifted design. An IRLS step does not depend on an affine change of the
        // columns, so the iterates are those of the raw design, and the stopping rule
        // below reads the raw intercept.
        beta[0] = design.shifted_from_raw(beta);

        let mut eta_now = column_buffer(eta, n);
        let mut prob_now = column_buffer(mu, n);
        let mut eta_next = column_buffer(search_eta, n);
        let mut prob_next = column_buffer(search_mu, n);
        let wz = column_buffer(weighted_response, n);
        let irls = column_buffer(irls_weights, n);
        let cross = column_buffer(cross, p);
        let candidate = column_buffer(candidate, p);

        let mut predict = |coefficients: &[f64], eta: &mut [f64], prob: &mut [f64]| {
            design.linear_predictor(coefficients[0], coefficients, dense_slopes, eta);
            for (e, q) in eta.iter_mut().zip(prob.iter_mut()) {
                *e = e.clamp(-LOGIT_ETA_CLAMP, LOGIT_ETA_CLAMP);
                *q = 1.0 / (1.0 + (-*e).exp());
            }
        };
        predict(beta, eta_now, prob_now);
        let mut deviance = logit_deviance(y, prob_now, weights);

        for iteration in 0..options.max_iter {
            for i in 0..n {
                let variance = (prob_now[i] * (1.0 - prob_now[i])).max(options.min_weight);
                irls[i] = variance * weights[i];
                wz[i] = irls[i] * (eta_now[i] + (y[i] - prob_now[i]) / variance);
            }
            design.weighted_gram(irls, products);
            let mut information = uncentred_information(products, p);
            add_ridge_to_diagonal(&mut information, ridge, true);
            let factor = factor_checked(&information)?;
            design.weighted_sums(wz, dense_acc, cross);
            let solved = solve_checked(&factor, &column_vector(cross))?;
            let full_step = solved.col_as_slice(0);

            let mut halvings = 0_usize;
            let next_deviance = loop {
                let scale = 0.5_f64.powi(i32::try_from(halvings).unwrap_or(i32::MAX));
                for (j, slot) in candidate.iter_mut().enumerate() {
                    *slot = beta[j] + (full_step[j] - beta[j]) * scale;
                }
                predict(candidate, eta_next, prob_next);
                let trial = logit_deviance(y, prob_next, weights);
                if trial.is_finite() && trial <= deviance {
                    break trial;
                }
                halvings += 1;
                if halvings > options.max_step_halvings {
                    // No downhill step exists here: a stationary point, as in the dense fit.
                    return self.logit_result(beta, iteration + 1);
                }
            };

            let converged_deviance = (next_deviance - deviance).abs() / (next_deviance.abs() + 0.1)
                < options.deviance_tolerance;
            let raw_intercept_change = (candidate[0] - beta[0])
                - (1..p)
                    .map(|j| (candidate[j] - beta[j]) * design.shifts[j])
                    .sum::<f64>();
            let change = (1..p)
                .map(|j| (candidate[j] - beta[j]).abs())
                .fold(raw_intercept_change.abs(), f64::max);

            beta.copy_from_slice(candidate);
            std::mem::swap(&mut eta_now, &mut eta_next);
            std::mem::swap(&mut prob_now, &mut prob_next);
            deviance = next_deviance;

            if converged_deviance || change < options.tolerance {
                return self.logit_result(beta, iteration + 1);
            }
        }
        Err(PreparedFitError::NonConvergence)
    }

    fn logit_result(
        &self,
        shifted: &mut [f64],
        iterations: usize,
    ) -> Result<(Mat<f64>, usize), PreparedFitError> {
        if shifted.iter().any(|value| !value.is_finite()) {
            return Err(PreparedFitError::NonFinite);
        }
        shifted[0] = self.all.raw_from_shifted(shifted[0], shifted);
        Ok((column_vector(shifted), iterations))
    }

    #[allow(clippy::too_many_lines)]
    fn fit_gamma_part(
        &self,
        weights: &[f64],
        warm: &Mat<f64>,
        ridge: f64,
        options: FitOptions,
        scratch: &mut PreparedScratch,
    ) -> Result<(Mat<f64>, usize), PreparedFitError> {
        let design = &self.positive;
        let (n, p) = (design.n, design.p);
        let y = &self.y_positive;
        let PreparedScratch {
            products,
            means,
            dense_slopes,
            dense_acc,
            cross,
            beta,
            candidate,
            eta,
            weighted_response,
            positive_weights,
            ..
        } = scratch;
        warm_values(warm, p, beta)?;

        positive_weights.clear();
        positive_weights.extend(self.positive_rows.iter().map(|&i| weights[i]));
        design.weighted_gram(positive_weights, products);
        let mut information = centred_information(products, p, means)?;
        add_ridge_to_diagonal(&mut information, ridge, true);
        let factor = factor_checked(&information)?;

        beta[0] = design.shifted_from_raw(beta) + (1..p).map(|j| beta[j] * means[j]).sum::<f64>();
        let shifted_intercept = |centred: f64, slopes: &[f64]| {
            centred - (1..p).map(|j| slopes[j] * means[j]).sum::<f64>()
        };

        let eta = column_buffer(eta, n);
        let wz = column_buffer(weighted_response, n);
        let cross = column_buffer(cross, p);
        let candidate = column_buffer(candidate, p);

        for iteration in 0..options.max_iter {
            design.linear_predictor(shifted_intercept(beta[0], beta), beta, dense_slopes, eta);
            for (((slot, &e), &yi), &w) in wz
                .iter_mut()
                .zip(eta.iter())
                .zip(y)
                .zip(positive_weights.iter())
            {
                let clamped = e.clamp(-GAMMA_LOG_LINK_ETA_CLAMP, GAMMA_LOG_LINK_ETA_CLAMP);
                let mean = clamped.exp();
                *slot = w * (clamped + (yi - mean) / mean);
            }
            design.weighted_sums(wz, dense_acc, cross);
            let total = cross[0];
            let rhs = Mat::from_fn(p, 1, |j, _| {
                if j == 0 {
                    total
                } else {
                    cross[j] - means[j] * total
                }
            });
            let solved = solve_checked(&factor, &rhs)?;
            candidate.copy_from_slice(solved.col_as_slice(0));

            let change = candidate
                .iter()
                .zip(beta.iter())
                .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
            beta.copy_from_slice(candidate);
            if change < options.tolerance {
                let shifted = shifted_intercept(beta[0], beta);
                design.linear_predictor(shifted, beta, dense_slopes, eta);
                let max_eta = eta.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
                if max_eta >= GAMMA_LOG_LINK_ETA_CLAMP {
                    return Err(PreparedFitError::Saturated);
                }
                beta[0] = design.raw_from_shifted(shifted, beta);
                return Ok((column_vector(beta), iteration + 1));
            }
        }
        Err(PreparedFitError::NonConvergence)
    }
}

/// Weighted binomial deviance on slices, as the dense logit computes it.
fn logit_deviance(y: &[f64], prob: &[f64], weights: &[f64]) -> f64 {
    const EPS: f64 = 1e-12;
    let mut deviance = 0.0;
    for ((&yi, &pi), &wi) in y.iter().zip(prob).zip(weights) {
        let clamped = pi.clamp(EPS, 1.0 - EPS);
        let outcome = f64::from(yi > 0.0);
        let term = outcome.mul_add(clamped.ln(), (1.0 - outcome) * (1.0 - clamped).ln());
        deviance = wi.mul_add(-2.0 * term, deviance);
    }
    deviance
}

/// A design for averaging fitted means over its rows, with coefficients from any fit.
///
/// Constant columns are allowed here, since nothing is centred.
#[derive(Debug, Clone)]
pub struct PreparedPredictionDesign {
    rows: CompactRows,
}

impl PreparedPredictionDesign {
    /// Read `x`, or `None` when column 0 is not a constant 1, there are more than
    /// [`MAX_COLUMNS`] columns, there are no rows, or the columns are not contiguous in
    /// memory.
    #[must_use]
    pub fn new(x: MatRef<'_, f64>) -> Option<Self> {
        Some(Self {
            rows: CompactRows::new(x)?,
        })
    }

    /// Rows of the design.
    #[must_use]
    pub const fn nrows(&self) -> usize {
        self.rows.n
    }

    /// The mean of `exp(x * beta)` over the rows, with the linear predictor clamped as in
    /// [`TweedieModel::predict_mean`](crate::TweedieModel::predict_mean): `Σ w_i m_i / Σ w_i`
    /// over the rows with `w_i != 0`. `beta` is on the raw design.
    ///
    /// # Panics
    ///
    /// Panics if `weights` is not as long as the design has rows, or `beta` is shorter
    /// than the design has columns.
    #[must_use]
    pub fn mean_gamma(&self, beta: &Mat<f64>, weights: &[f64]) -> f64 {
        assert_eq!(weights.len(), self.rows.n, "one weight per row");
        let slopes = beta.col_as_slice(0);
        let intercept = self.rows.shifted_from_raw(slopes);
        let mut dense_slopes = Vec::new();
        self.rows.select_dense(slopes, &mut dense_slopes);
        let (mut sum, mut total) = (0.0_f64, 0.0_f64);
        for (i, &weight) in weights.iter().enumerate() {
            if weight != 0.0 {
                let eta = self.rows.row_eta(i, intercept, slopes, &dense_slopes);
                sum += weight * exp_clamped(eta);
                total += weight;
            }
        }
        sum / total
    }

    /// The mean of the two-part prediction over the rows, as in
    /// [`TwoPartModel::predict_mean`](crate::TwoPartModel::predict_mean): `Σ w_i m_i / Σ w_i`
    /// over the rows with `w_i != 0`. The coefficients are on the raw design.
    ///
    /// # Panics
    ///
    /// Panics if `weights` is not as long as the design has rows, or a coefficient vector
    /// is shorter than the design has columns.
    #[must_use]
    pub fn mean_two_part(
        &self,
        beta_logit: &Mat<f64>,
        beta_gamma: &Mat<f64>,
        weights: &[f64],
    ) -> f64 {
        assert_eq!(weights.len(), self.rows.n, "one weight per row");
        let (logit, gamma) = (beta_logit.col_as_slice(0), beta_gamma.col_as_slice(0));
        let (intercept_logit, intercept_gamma) = (
            self.rows.shifted_from_raw(logit),
            self.rows.shifted_from_raw(gamma),
        );
        let (mut dense_logit, mut dense_gamma) = (Vec::new(), Vec::new());
        self.rows.select_dense(logit, &mut dense_logit);
        self.rows.select_dense(gamma, &mut dense_gamma);
        let (mut sum, mut total) = (0.0_f64, 0.0_f64);
        for (i, &weight) in weights.iter().enumerate() {
            if weight != 0.0 {
                let eta_logit = self.rows.row_eta(i, intercept_logit, logit, &dense_logit);
                let eta_gamma = self.rows.row_eta(i, intercept_gamma, gamma, &dense_gamma);
                let prob = 1.0 / (1.0 + (-eta_logit).exp());
                let mean = eta_gamma
                    .clamp(-GAMMA_LOG_LINK_ETA_CLAMP, GAMMA_LOG_LINK_ETA_CLAMP)
                    .exp();
                sum += weight * (prob * mean);
                total += weight;
            }
        }
        sum / total
    }
}
