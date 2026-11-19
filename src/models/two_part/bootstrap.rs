//! Row-resampling bootstrap and its percentile summaries.

use epistat_inference::interval::{Percentile, percentile};
use epistat_rng::Stream;
use faer::Mat;

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

    let mut rng = Stream::seeded(bootstrap.seed);
    let mut betas_logit = Vec::with_capacity(bootstrap.iterations);
    let mut betas_gamma = Vec::with_capacity(bootstrap.iterations);
    let mut failures = 0;

    for _ in 0..bootstrap.iterations {
        let indices = (0..x.nrows())
            .map(|_| rng.below(x.nrows()))
            .collect::<Vec<_>>();
        let x_sample = select_rows(x, &indices);
        let y_sample = select_values(y, &indices);
        match fit_two_part(x_sample.as_ref(), y_sample.as_ref(), options) {
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

/// Percentile confidence intervals from bootstrap draws, one per coefficient.
///
/// The bounds are R's type 7 quantiles (`quantile(x, c(alpha / 2, 1 - alpha / 2), type = 7)`)
/// of the draws, so they can fall between two draws. A coefficient whose draws contain a
/// NaN, or an `alpha` outside `(0, 1)`, gives NaN bounds.
#[must_use]
pub fn bootstrap_percentile_ci(betas: &[Mat<f64>], alpha: f64) -> Vec<ConfidenceInterval> {
    if betas.is_empty() {
        return Vec::new();
    }
    let n = betas[0].nrows();
    let mut intervals = Vec::with_capacity(n);
    let mut values: Vec<f64> = Vec::with_capacity(betas.len());
    for col in 0..n {
        values.clear();
        values.extend(betas.iter().map(|b| b[(col, 0)]));
        let (lower, upper) =
            percentile(&mut values, alpha, Percentile::Type7).unwrap_or((f64::NAN, f64::NAN));
        intervals.push(ConfidenceInterval { lower, upper });
    }
    intervals
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
