//! Row-resampling bootstrap and its percentile summaries.

use faer::Mat;
use rand::prelude::*;

use super::{
    BootstrapOptions, BootstrapResult, BootstrapSummary, ConfidenceInterval, FitOptions,
    TwoPartError, fit_two_part,
};
use crate::models::matrix_ops::{select_rows, select_values};
use crate::utils::{mean_vector, std_vector};

/// Bootstrap two-part model fits by resampling rows with replacement.
///
/// # Errors
///
/// Returns `TwoPartError` if inputs are malformed or the bootstrap fails.
pub fn bootstrap(
    x: &Mat<f64>,
    y: &Mat<f64>,
    options: FitOptions,
    bootstrap: BootstrapOptions,
) -> Result<BootstrapResult, TwoPartError> {
    if bootstrap.iterations == 0 {
        return Err(TwoPartError::InvalidBootstrapIterations);
    }
    if x.nrows() == 0 {
        return Err(TwoPartError::EmptyBootstrapSample);
    }
    if x.nrows() != y.nrows() {
        return Err(TwoPartError::DimensionMismatch {
            rows: x.nrows(),
            len: y.nrows(),
        });
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(bootstrap.seed);
    let mut betas_logit = Vec::with_capacity(bootstrap.iterations);
    let mut betas_gamma = Vec::with_capacity(bootstrap.iterations);
    let mut failures = 0;

    for _ in 0..bootstrap.iterations {
        let indices = (0..x.nrows())
            .map(|_| rng.random_range(0..x.nrows()))
            .collect::<Vec<_>>();
        let x_sample = select_rows(x, &indices);
        let y_sample = select_values(y, &indices);
        match fit_two_part(&x_sample, &y_sample, options) {
            Ok((model, _)) => {
                betas_logit.push(model.beta_logit);
                betas_gamma.push(model.beta_gamma);
            }
            Err(TwoPartError::NonConvergence) if bootstrap.skip_nonconvergence => {
                failures += 1;
                if failures > bootstrap.max_failures {
                    return Err(TwoPartError::TooManyBootstrapFailures(failures));
                }
            }
            Err(err) => return Err(err),
        }
    }

    Ok(BootstrapResult {
        betas_logit,
        betas_gamma,
        failures,
    })
}

/// Percentile confidence intervals from bootstrap draws.
#[must_use]
pub fn bootstrap_percentile_ci(betas: &[Mat<f64>], alpha: f64) -> Vec<ConfidenceInterval> {
    if betas.is_empty() {
        return Vec::new();
    }
    let n = betas[0].nrows();
    let last = betas.len().saturating_sub(1);
    let (mut lower_idx, mut upper_idx) = crate::utils::boot_index_bounds(alpha, betas.len());
    lower_idx = lower_idx.min(last);
    upper_idx = upper_idx.min(last).max(lower_idx);

    let mut intervals = Vec::with_capacity(n);
    let mut values: Vec<f64> = Vec::with_capacity(betas.len());
    for col in 0..n {
        values.clear();
        values.extend(betas.iter().map(|b| b[(col, 0)]));
        let upper = order_statistic_unsorted(&mut values, upper_idx).unwrap_or(f64::NAN);
        let lower = if lower_idx == upper_idx {
            upper
        } else {
            order_statistic_unsorted(&mut values[..=upper_idx], lower_idx).unwrap_or(upper)
        };
        intervals.push(ConfidenceInterval { lower, upper });
    }
    intervals
}

fn order_statistic_unsorted(values: &mut [f64], nth: usize) -> Option<f64> {
    if values.is_empty() || nth >= values.len() {
        return None;
    }
    values.select_nth_unstable_by(nth, f64::total_cmp);
    Some(values[nth])
}

/// Compute bootstrap mean, SE, and percentile CI.
#[must_use]
pub fn bootstrap_summary(betas: &[Mat<f64>], alpha: f64) -> BootstrapSummary {
    if betas.is_empty() {
        return BootstrapSummary {
            mean: Mat::zeros(0, 1),
            se: Mat::zeros(0, 1),
            ci: Vec::new(),
        };
    }
    let mean = mean_vector(betas);
    let se = std_vector(betas, &mean);
    let ci = bootstrap_percentile_ci(betas, alpha);
    BootstrapSummary { mean, se, ci }
}
