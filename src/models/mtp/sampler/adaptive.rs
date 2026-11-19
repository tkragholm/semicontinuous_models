//! Adaptive random-walk proposal for the fixed-effect block.
//!
//! The block holds `alpha`, `beta` and `log(omega_sq)`. In the marginalized model the
//! log-scale location of the positive part is `xi = x'beta - log(p) - omega_sq / 2`
//! (plus a `kappa` term under the log-skew-normal part), with `p` the logistic
//! probability from `x'alpha`. The positive outcomes pin `xi` down closely, so the
//! posterior correlates the binary and mean coefficients with each other and with
//! `omega_sq`, and the block is updated jointly.
//!
//! # Algorithm
//!
//! The proposal is `theta' = theta + lambda * L * z` with `z ~ N(0, I)` and `L` the
//! lower Cholesky factor of a covariance `C`. `C` starts as a block-diagonal
//! approximation of the posterior covariance and is replaced during burn-in by the
//! empirical covariance of the chain in two windows, as in the adaptive Metropolis
//! algorithm of Haario, Saksman and Tamminen (2001) with the windowed estimation of
//! Stan. After each replacement `lambda` restarts at `2.38 / sqrt(d)` and is then
//! tuned towards the acceptance band. Adaptation stops at the end of burn-in, so the
//! retained draws come from a fixed Metropolis kernel.
//!
//! The estimate in a window of `n` draws is shrunk towards a small multiple of the
//! diagonal of the starting covariance,
//! `C = n / (n + 5) * S + 5 / (n + 5) * 1e-3 * diag(C0)`, which keeps `C` positive
//! definite when the chain has visited few distinct points.
//!
//! # References
//!
//! - Haario, H., Saksman, E. and Tamminen, J. (2001). An adaptive Metropolis
//!   algorithm. Bernoulli 7(2), 223-242.
//! - Roberts, G. O., Gelman, A. and Gilks, W. R. (1997). Weak convergence and optimal
//!   scaling of random walk Metropolis algorithms. Annals of Applied Probability 7(1),
//!   110-120.

use epistat_rng::Stream;
use faer::Mat;

use super::math::{
    adapt_scalar_scale, cholesky_lower, sample_standard_normal,
    usize_to_f64_reexport as usize_to_f64,
};
use crate::models::mtp::types::MtpProposalTuning;

/// Random-walk scale numerator of Roberts, Gelman and Gilks (1997): `2.38 / sqrt(d)`
/// is optimal for a `d`-dimensional Gaussian target whose covariance the proposal
/// matches.
pub(super) const RANDOM_WALK_SCALE: f64 = 2.38;

const SHRINKAGE_DRAWS: f64 = 5.0;
const SHRINKAGE_TARGET: f64 = 1.0e-3;
/// Minimum window length per block dimension before the empirical covariance is used.
const MIN_WINDOW_DRAWS_PER_DIMENSION: usize = 10;

#[derive(Debug, Clone)]
pub(super) struct BlockProposal {
    initial_diagonal: Vec<f64>,
    factor: Mat<f64>,
    scale: f64,
    adaptations: usize,
    min_variance: f64,
    window: CovarianceWindow,
}

impl BlockProposal {
    /// Proposal with starting covariance `initial`, whose diagonal is floored at
    /// `min_draw_scale^2`.
    pub(super) fn new(mut initial: Mat<f64>, min_draw_scale: f64) -> Self {
        let dim = initial.ncols();
        let min_variance = min_draw_scale * min_draw_scale;
        for index in 0..dim {
            initial[(index, index)] = initial[(index, index)].max(min_variance);
        }
        let initial_diagonal = (0..dim).map(|index| initial[(index, index)]).collect();
        let factor = cholesky_lower(&initial).unwrap_or_else(|| diagonal_factor(&initial));
        Self {
            initial_diagonal,
            factor,
            scale: default_scale(dim),
            adaptations: 0,
            min_variance,
            window: CovarianceWindow::new(dim),
        }
    }

    pub(super) const fn dim(&self) -> usize {
        self.initial_diagonal.len()
    }

    #[cfg(test)]
    pub(super) const fn scale(&self) -> f64 {
        self.scale
    }

    #[cfg(test)]
    pub(super) const fn set_scale(&mut self, scale: f64) {
        self.scale = scale;
    }

    /// Robbins-Monro step of `lambda` for the acceptance rate of the last interval. The
    /// step count restarts with `lambda` whenever the covariance is replaced.
    pub(super) fn adapt_scale(&mut self, acceptance: f64, tuning: MtpProposalTuning) {
        self.adaptations += 1;
        self.scale = adapt_scalar_scale(self.scale, acceptance, self.adaptations, tuning);
    }

    /// The proposal's lower Cholesky factor and `lambda`, for checks that it is fixed.
    #[cfg(test)]
    pub(super) fn snapshot_into(&self, out: &mut Vec<f64>) {
        out.extend(self.factor.col_iter().flat_map(|col| col.iter().copied()));
        out.push(self.scale);
    }

    /// Fill `step` with `lambda * L * z`.
    pub(super) fn draw_step_into(&self, step: &mut Vec<f64>, rng: &mut Stream) {
        let dim = self.dim();
        step.clear();
        step.extend((0..dim).map(|_| sample_standard_normal(rng)));
        for row in (0..dim).rev() {
            let value = (0..=row)
                .map(|col| self.factor[(row, col)] * step[col])
                .sum::<f64>();
            step[row] = self.scale * value;
        }
    }

    /// Add the current block value to the covariance window.
    pub(super) fn record(&mut self, theta: &[f64]) {
        self.window.push(theta);
    }

    /// Replace the proposal covariance by the shrunk estimate from the current window
    /// and restart the window. The covariance is kept when the window is too short or
    /// the estimate is not positive definite. Returns whether it was replaced.
    pub(super) fn refresh_from_window(&mut self) -> bool {
        let dim = self.dim();
        let count = self.window.count;
        let refreshed = if count >= MIN_WINDOW_DRAWS_PER_DIMENSION * dim {
            let n = usize_to_f64(count);
            let weight = n / (n + SHRINKAGE_DRAWS);
            let mut covariance = self.window.covariance();
            for row in 0..dim {
                for col in 0..dim {
                    covariance[(row, col)] *= weight;
                }
                covariance[(row, row)] = (1.0 - weight)
                    .mul_add(
                        SHRINKAGE_TARGET * self.initial_diagonal[row],
                        covariance[(row, row)],
                    )
                    .max(self.min_variance);
            }
            cholesky_lower(&covariance).is_some_and(|factor| {
                self.factor = factor;
                self.scale = default_scale(dim);
                self.adaptations = 0;
                true
            })
        } else {
            false
        };
        self.window = CovarianceWindow::new(dim);
        refreshed
    }
}

fn default_scale(dim: usize) -> f64 {
    RANDOM_WALK_SCALE / usize_to_f64(dim.max(1)).sqrt()
}

fn diagonal_factor(covariance: &Mat<f64>) -> Mat<f64> {
    let dim = covariance.ncols();
    Mat::from_fn(dim, dim, |row, col| {
        if row == col {
            covariance[(row, row)].max(0.0).sqrt()
        } else {
            0.0
        }
    })
}

/// Running mean and sum of squared deviations (Welford's algorithm).
#[derive(Debug, Clone)]
struct CovarianceWindow {
    count: usize,
    mean: Vec<f64>,
    deviations: Mat<f64>,
    before: Vec<f64>,
}

impl CovarianceWindow {
    fn new(dim: usize) -> Self {
        Self {
            count: 0,
            mean: vec![0.0; dim],
            deviations: Mat::zeros(dim, dim),
            before: vec![0.0; dim],
        }
    }

    fn push(&mut self, theta: &[f64]) {
        self.count += 1;
        let n = usize_to_f64(self.count);
        for ((before, mean), value) in self.before.iter_mut().zip(&mut self.mean).zip(theta) {
            *before = value - *mean;
            *mean += *before / n;
        }
        for (row, (value, mean)) in theta.iter().zip(&self.mean).enumerate() {
            let after_row = value - mean;
            for (col, before) in self.before.iter().enumerate() {
                self.deviations[(row, col)] =
                    before.mul_add(after_row, self.deviations[(row, col)]);
            }
        }
    }

    fn covariance(&self) -> Mat<f64> {
        let dim = self.mean.len();
        let denominator = usize_to_f64(self.count.saturating_sub(1).max(1));
        Mat::from_fn(dim, dim, |row, col| {
            0.5 * (self.deviations[(row, col)] + self.deviations[(col, row)]) / denominator
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_covariance_matches_two_pass_estimate() {
        let draws = [[1.0, 2.0], [2.0, 1.5], [0.5, 3.0], [1.5, 2.5], [3.0, 0.5]];
        let mut window = CovarianceWindow::new(2);
        for draw in &draws {
            window.push(draw);
        }
        let n = usize_to_f64(draws.len());
        let mean = [
            draws.iter().map(|d| d[0]).sum::<f64>() / n,
            draws.iter().map(|d| d[1]).sum::<f64>() / n,
        ];
        let covariance = window.covariance();
        for row in 0..2 {
            for col in 0..2 {
                let expected = draws
                    .iter()
                    .map(|d| (d[row] - mean[row]) * (d[col] - mean[col]))
                    .sum::<f64>()
                    / (n - 1.0);
                assert!((covariance[(row, col)] - expected).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn step_has_the_proposal_covariance() {
        let covariance = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => 4.0,
            (1, 1) => 1.0,
            _ => 1.2,
        });
        let mut proposal = BlockProposal::new(covariance.clone(), 1.0e-3);
        proposal.set_scale(1.0);
        let mut rng = Stream::seeded(7);
        let mut window = CovarianceWindow::new(2);
        let mut step = Vec::new();
        for _ in 0..200_000 {
            proposal.draw_step_into(&mut step, &mut rng);
            window.push(&step);
        }
        let estimate = window.covariance();
        for row in 0..2 {
            for col in 0..2 {
                assert!((estimate[(row, col)] - covariance[(row, col)]).abs() < 0.05);
            }
        }
    }

    #[test]
    fn short_window_keeps_the_covariance() {
        let mut proposal = BlockProposal::new(Mat::identity(3, 3), 1.0e-3);
        proposal.set_scale(0.5);
        proposal.record(&[0.0, 1.0, 2.0]);
        assert!(!proposal.refresh_from_window());
        assert!((proposal.scale() - 0.5).abs() < f64::EPSILON);
    }
}
