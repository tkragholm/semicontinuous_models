//! MCMC and posterior predictive diagnostics for MTP.

use crate::input::LongitudinalModelInput;
use crate::utils::{
    CalibrationBinSummary, EffectIntervalSummary, calibration_bins_summary, dot_row,
    summarize_draws_in_place, usize_to_f64,
};

use super::likelihood::logistic_stable;

use super::posterior::MtpPosteriorSamples;
use super::types::{
    MtpConvergenceSummary, MtpError, validate_chain_count, validate_draw_dimensions,
};

/// Lag-`k` autocorrelation for a scalar chain.
#[must_use]
pub fn autocorrelation(series: &[f64], lag: usize) -> f64 {
    if series.is_empty() || lag >= series.len() {
        return 0.0;
    }

    let mean = series.iter().sum::<f64>() / usize_to_f64(series.len());
    let denominator = sum_squared_deviations(series, mean);
    if denominator <= 0.0 {
        return 0.0;
    }

    autocorrelation_with_stats(series, lag, mean, denominator)
}

/// Sum of squared deviations from `mean`, i.e. the lag-0 autocovariance numerator.
fn sum_squared_deviations(series: &[f64], mean: f64) -> f64 {
    series.iter().fold(0.0, |acc, value| {
        let centered = value - mean;
        centered.mul_add(centered, acc)
    })
}

/// Lag-`k` autocorrelation given a precomputed `mean` and lag-0 `denominator`.
///
/// The denominator (sum of squared deviations) is identical for every lag of a
/// given series, so callers that sweep many lags should compute it once and
/// reuse it here instead of paying an O(n) pass per lag.
fn autocorrelation_with_stats(series: &[f64], lag: usize, mean: f64, denominator: f64) -> f64 {
    let n = series.len() - lag;
    let mut numerator = 0.0;
    for idx in 0..n {
        numerator = (series[idx] - mean).mul_add(series[idx + lag] - mean, numerator);
    }
    numerator / denominator
}

/// Heuristic effective sample size using positive autocorrelation truncation.
#[must_use]
pub fn effective_sample_size(series: &[f64]) -> f64 {
    let n = series.len();
    if n < 2 {
        return usize_to_f64(n);
    }

    let mean = series.iter().sum::<f64>() / usize_to_f64(n);
    let denominator = sum_squared_deviations(series, mean);
    if denominator <= 0.0 {
        return usize_to_f64(n);
    }

    let mut rho_sum = 0.0;
    for lag in 1..n {
        let rho = autocorrelation_with_stats(series, lag, mean, denominator);
        if rho <= 0.0 {
            break;
        }
        rho_sum += rho;
    }

    usize_to_f64(n) / (2.0f64.mul_add(rho_sum, 1.0)).max(1.0)
}

/// Summarize split-R-hat convergence diagnostics across posterior chains.
///
/// This implementation:
/// - requires at least two chains,
/// - truncates all chains to the same minimum even draw count,
/// - computes split-R-hat for all scalar parameters in `alpha`, `beta`, `kappa`, and `omega_sq`.
///
/// # Errors
///
/// Returns `MtpError` if chain counts/draw lengths are insufficient or dimensions mismatch.
pub fn summarize_multi_chain_convergence(
    chains: &[MtpPosteriorSamples],
) -> Result<MtpConvergenceSummary, MtpError> {
    validate_chain_count(chains.len(), 2)?;

    let min_draws = chains
        .iter()
        .map(MtpPosteriorSamples::len)
        .min()
        .unwrap_or(0);
    let draws_per_chain_used = min_draws - (min_draws % 2);
    if draws_per_chain_used < 4 {
        return Err(MtpError::InsufficientChainDraws {
            minimum: 4,
            found: draws_per_chain_used,
        });
    }

    let first_draw = chains.first().and_then(|chain| chain.draws.first()).ok_or(
        MtpError::InsufficientChainDraws {
            minimum: 4,
            found: 0,
        },
    )?;
    let alpha_len = first_draw.alpha.len();
    let beta_len = first_draw.beta.len();

    for chain in chains {
        if chain.len() < draws_per_chain_used {
            return Err(MtpError::InsufficientChainDraws {
                minimum: draws_per_chain_used,
                found: chain.len(),
            });
        }
        for draw in chain.draws.iter().take(draws_per_chain_used) {
            validate_draw_dimensions(alpha_len, beta_len, draw.alpha.len(), draw.beta.len())
                .map_err(|_| MtpError::InconsistentPosteriorDimensions)?;
        }
    }

    let alpha_split_rhat = (0..alpha_len)
        .map(|index| split_rhat_from_chains(chains, draws_per_chain_used, |draw| draw.alpha[index]))
        .collect::<Result<Vec<_>, _>>()?;
    let beta_split_rhat = (0..beta_len)
        .map(|index| split_rhat_from_chains(chains, draws_per_chain_used, |draw| draw.beta[index]))
        .collect::<Result<Vec<_>, _>>()?;
    let kappa_split_rhat =
        split_rhat_from_chains(chains, draws_per_chain_used, |draw| draw.kappa).ok();
    let omega_sq_split_rhat =
        split_rhat_from_chains(chains, draws_per_chain_used, |draw| draw.omega_sq).ok();

    let max_split_rhat = alpha_split_rhat
        .iter()
        .copied()
        .chain(beta_split_rhat.iter().copied())
        .chain(kappa_split_rhat)
        .chain(omega_sq_split_rhat)
        .max_by(f64::total_cmp);

    Ok(MtpConvergenceSummary {
        chain_count: chains.len(),
        draws_per_chain_used,
        alpha_split_rhat,
        beta_split_rhat,
        kappa_split_rhat,
        omega_sq_split_rhat,
        max_split_rhat,
    })
}

/// Posterior predictive diagnostics summary.
#[derive(Debug, Clone, Default)]
pub struct PosteriorPredictiveSummary {
    pub observed_zero_rate: f64,
    pub observed_mean: f64,
    pub predicted_zero_rate: EffectIntervalSummary,
    pub predicted_mean: EffectIntervalSummary,
    pub brier_score: EffectIntervalSummary,
    pub calibration_bins: Vec<CalibrationBinSummary>,
}

/// Compute posterior predictive diagnostics from retained draws.
///
/// Diagnostics include:
/// - posterior interval for average zero-rate,
/// - posterior interval for average mean outcome,
/// - posterior interval for Brier score of positive-probability predictions,
/// - calibration bins comparing posterior-mean predicted probability against observed rates.
///
/// # Errors
///
/// Returns `MtpError` if inputs/draws are invalid.
pub fn posterior_predictive_summary(
    samples: &MtpPosteriorSamples,
    input: &LongitudinalModelInput,
    calibration_bins: usize,
) -> Result<PosteriorPredictiveSummary, MtpError> {
    if samples.is_empty() {
        return Err(MtpError::EmptyPosterior);
    }
    if calibration_bins == 0 {
        return Err(MtpError::InvalidCalibrationBins);
    }
    input.validate()?;

    let alpha_len = samples.draws.first().map_or(0, |draw| draw.alpha.len());
    let beta_len = samples.draws.first().map_or(0, |draw| draw.beta.len());

    if input.x_binary.ncols() != alpha_len {
        return Err(MtpError::DesignCoefficientMismatch {
            design_cols: input.x_binary.ncols(),
            coef_len: alpha_len,
        });
    }
    if input.x_mean.ncols() != beta_len {
        return Err(MtpError::DesignCoefficientMismatch {
            design_cols: input.x_mean.ncols(),
            coef_len: beta_len,
        });
    }

    let nrows = input.outcome.nrows();
    let mut predicted_probability_sum = vec![0.0; nrows];
    let mut predicted_zero_rate_draws: Vec<f64> = Vec::with_capacity(samples.len());
    let mut predicted_mean_draws: Vec<f64> = Vec::with_capacity(samples.len());
    let mut brier_draws: Vec<f64> = Vec::with_capacity(samples.len());

    for draw in &samples.draws {
        validate_draw_dimensions(alpha_len, beta_len, draw.alpha.len(), draw.beta.len())?;

        let mut zero_sum = 0.0;
        let mut mean_sum = 0.0;
        let mut brier_sum = 0.0;

        for (row, probability_sum) in predicted_probability_sum.iter_mut().enumerate() {
            let p_positive = logistic_stable(dot_row(&input.x_binary, row, &draw.alpha));
            // MTP beta parameterizes the marginal mean directly.
            let expected = dot_row(&input.x_mean, row, &draw.beta).exp();
            let observed_positive = if input.outcome[(row, 0)] > 0.0 {
                1.0
            } else {
                0.0
            };

            *probability_sum += p_positive;
            zero_sum += 1.0 - p_positive;
            mean_sum += expected;
            brier_sum =
                (p_positive - observed_positive).mul_add(p_positive - observed_positive, brier_sum);
        }

        predicted_zero_rate_draws.push(zero_sum / usize_to_f64(nrows));
        predicted_mean_draws.push(mean_sum / usize_to_f64(nrows));
        brier_draws.push(brier_sum / usize_to_f64(nrows));
    }

    let observed_zero_count = (0..nrows)
        .filter(|&row| input.outcome[(row, 0)] <= 0.0)
        .count();
    let observed_zero_rate = usize_to_f64(observed_zero_count) / usize_to_f64(nrows);
    let observed_mean =
        (0..nrows).map(|row| input.outcome[(row, 0)]).sum::<f64>() / usize_to_f64(nrows);

    let predicted_probability_mean = predicted_probability_sum
        .into_iter()
        .map(|value| value / usize_to_f64(samples.len()))
        .collect::<Vec<_>>();

    Ok(PosteriorPredictiveSummary {
        observed_zero_rate,
        observed_mean,
        predicted_zero_rate: summarize_draws_in_place(&mut predicted_zero_rate_draws),
        predicted_mean: summarize_draws_in_place(&mut predicted_mean_draws),
        brier_score: summarize_draws_in_place(&mut brier_draws),
        calibration_bins: calibration_bins_summary(
            &predicted_probability_mean,
            input,
            calibration_bins,
        ),
    })
}

fn split_rhat_from_chains<F>(
    chains: &[MtpPosteriorSamples],
    draws_per_chain_used: usize,
    extractor: F,
) -> Result<f64, MtpError>
where
    F: Fn(&super::posterior::MtpPosteriorDraw) -> f64,
{
    if chains.len() < 2 || draws_per_chain_used < 4 || !draws_per_chain_used.is_multiple_of(2) {
        return Err(MtpError::InsufficientChainDraws {
            minimum: 4,
            found: draws_per_chain_used,
        });
    }

    let half = draws_per_chain_used / 2;
    // Accumulate only the per-split-chain (mean, variance) moments in a single
    // pass, rather than materializing 2·`chains.len()` value vectors of length
    // `half` for every parameter.
    let mut means = Vec::with_capacity(chains.len() * 2);
    let mut variances = Vec::with_capacity(chains.len() * 2);
    for chain in chains {
        let (first_mean, first_var, _) =
            mean_and_sample_variance(chain.draws.iter().take(half).map(&extractor));
        let (second_mean, second_var, _) =
            mean_and_sample_variance(chain.draws.iter().skip(half).take(half).map(&extractor));
        means.push(first_mean);
        variances.push(first_var);
        means.push(second_mean);
        variances.push(second_var);
    }

    split_rhat_from_moments(&means, &variances, half)
}

/// Streaming mean and sample variance (Welford, `n - 1` denominator) returning
/// `(mean, variance, count)`. Single pass, no allocation.
fn mean_and_sample_variance<I>(values: I) -> (f64, f64, usize)
where
    I: Iterator<Item = f64>,
{
    let mut count = 0usize;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for value in values {
        count += 1;
        let delta = value - mean;
        mean += delta / usize_to_f64(count);
        let delta2 = value - mean;
        m2 = delta.mul_add(delta2, m2);
    }
    let variance = if count < 2 {
        0.0
    } else {
        m2 / usize_to_f64(count - 1)
    };
    (mean, variance, count)
}

/// Split-R-hat from per-split-chain means and variances (each split chain has
/// `n` draws). Equivalent to the prior `split_rhat_scalar`, but fed precomputed
/// moments instead of value vectors.
fn split_rhat_from_moments(means: &[f64], variances: &[f64], n: usize) -> Result<f64, MtpError> {
    let m = means.len();
    if m < 2 {
        return Err(MtpError::InvalidChainCount { min: 2, found: m });
    }
    if n < 2 {
        return Err(MtpError::InsufficientChainDraws {
            minimum: 2,
            found: n,
        });
    }

    let mean_of_means = means.iter().sum::<f64>() / usize_to_f64(m);
    let between = usize_to_f64(n)
        * means
            .iter()
            .map(|mean| {
                let centered = *mean - mean_of_means;
                centered * centered
            })
            .sum::<f64>()
        / usize_to_f64(m - 1);
    let within = variances.iter().sum::<f64>() / usize_to_f64(m);

    if !(within.is_finite() && within > 0.0 && between.is_finite()) {
        return Ok(1.0);
    }

    let n_f64 = usize_to_f64(n);
    let var_plus = ((n_f64 - 1.0) / n_f64).mul_add(within, between / n_f64);
    if !var_plus.is_finite() || var_plus <= 0.0 {
        return Ok(1.0);
    }

    Ok((var_plus / within).sqrt().max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::mtp::posterior::MtpPosteriorDraw;
    use faer::Mat;

    #[test]
    fn autocorrelation_is_zero_for_invalid_lag() {
        let values = [1.0, 2.0, 3.0];
        assert!((autocorrelation(&values, 3) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ess_bounded_by_chain_length() {
        let values = [1.0, 1.5, 2.0, 2.5, 3.0];
        let ess = effective_sample_size(&values);
        assert!(ess <= 5.0);
        assert!(ess > 0.0);
    }

    fn chain_from_alphas(alphas: &[f64]) -> MtpPosteriorSamples {
        MtpPosteriorSamples {
            draws: alphas
                .iter()
                .map(|&a| MtpPosteriorDraw {
                    alpha: vec![a],
                    beta: vec![0.0],
                    kappa: 0.1,
                    omega_sq: 1.0,
                })
                .collect(),
        }
    }

    #[test]
    fn split_rhat_is_one_for_identical_chains() {
        let chain = chain_from_alphas(&[0.0, 1.0, 0.0, 1.0]);
        let summary = summarize_multi_chain_convergence(&[chain.clone(), chain])
            .expect("two valid chains should summarize");
        assert_eq!(summary.alpha_split_rhat.len(), 1);
        assert!((summary.alpha_split_rhat[0] - 1.0).abs() < 1.0e-9);
        assert!((summary.max_split_rhat.unwrap() - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn split_rhat_is_finite_and_at_least_one_with_between_chain_variation() {
        let chain_a = chain_from_alphas(&[0.0, 1.0, 0.0, 1.0]);
        let chain_b = chain_from_alphas(&[5.0, 6.0, 5.0, 6.0]);
        let summary = summarize_multi_chain_convergence(&[chain_a, chain_b])
            .expect("two valid chains should summarize");
        let rhat = summary.alpha_split_rhat[0];
        assert!(rhat.is_finite());
        assert!(rhat >= 1.0);
        // Means differ sharply between chains, so R-hat must exceed 1.
        assert!(rhat > 1.0);
    }

    #[test]
    fn ppc_requires_draws() {
        let input = LongitudinalModelInput::new(
            Mat::from_fn(2, 1, |_i, _| 1.0),
            Mat::from_fn(2, 1, |_i, _| 1.0),
            Mat::from_fn(2, 1, |_i, _| 1.0),
            vec![1, 1],
            vec![0.0, 1.0],
        );

        let result = posterior_predictive_summary(&MtpPosteriorSamples::default(), &input, 10);
        assert!(matches!(result, Err(MtpError::EmptyPosterior)));
    }

    #[test]
    fn ppc_returns_calibration_bins() {
        let input = LongitudinalModelInput::new(
            Mat::from_fn(4, 1, |row, _| if row % 2 == 0 { 0.0 } else { 2.0 }),
            Mat::from_fn(
                4,
                2,
                |row, col| {
                    if col == 0 { 1.0 } else { usize_to_f64(row) }
                },
            ),
            Mat::from_fn(
                4,
                2,
                |row, col| {
                    if col == 0 { 1.0 } else { usize_to_f64(row) }
                },
            ),
            vec![1, 1, 2, 2],
            vec![0.0, 1.0, 0.0, 1.0],
        );

        let samples = MtpPosteriorSamples {
            draws: vec![
                MtpPosteriorDraw {
                    alpha: vec![0.0, 0.2],
                    beta: vec![0.1, 0.05],
                    kappa: 0.0,
                    omega_sq: 1.0,
                },
                MtpPosteriorDraw {
                    alpha: vec![0.0, 0.1],
                    beta: vec![0.2, 0.03],
                    kappa: 0.1,
                    omega_sq: 1.2,
                },
            ],
        };

        let summary =
            posterior_predictive_summary(&samples, &input, 5).expect("ppc summary should run");
        assert_eq!(summary.calibration_bins.len(), 5);
        let total_count: usize = summary.calibration_bins.iter().map(|bin| bin.count).sum();
        assert_eq!(total_count, input.outcome.nrows());
        assert!(summary.predicted_mean.mean.is_finite());
        assert!(summary.brier_score.mean >= 0.0);
    }

    #[test]
    fn convergence_requires_two_chains() {
        let chains = vec![MtpPosteriorSamples {
            draws: vec![
                MtpPosteriorDraw {
                    alpha: vec![0.1],
                    beta: vec![0.2],
                    kappa: 0.0,
                    omega_sq: 1.0,
                };
                4
            ],
        }];
        let result = summarize_multi_chain_convergence(&chains);
        assert!(matches!(result, Err(MtpError::InvalidChainCount { .. })));
    }

    #[test]
    fn convergence_returns_split_rhat_summary() {
        let chain_a = MtpPosteriorSamples {
            draws: vec![
                MtpPosteriorDraw {
                    alpha: vec![0.0, 0.5],
                    beta: vec![1.0],
                    kappa: 0.1,
                    omega_sq: 1.0,
                },
                MtpPosteriorDraw {
                    alpha: vec![0.1, 0.6],
                    beta: vec![1.1],
                    kappa: 0.2,
                    omega_sq: 1.1,
                },
                MtpPosteriorDraw {
                    alpha: vec![0.2, 0.7],
                    beta: vec![1.2],
                    kappa: 0.3,
                    omega_sq: 1.2,
                },
                MtpPosteriorDraw {
                    alpha: vec![0.3, 0.8],
                    beta: vec![1.3],
                    kappa: 0.4,
                    omega_sq: 1.3,
                },
            ],
        };
        let chain_b = MtpPosteriorSamples {
            draws: vec![
                MtpPosteriorDraw {
                    alpha: vec![0.05, 0.45],
                    beta: vec![1.05],
                    kappa: 0.15,
                    omega_sq: 1.05,
                },
                MtpPosteriorDraw {
                    alpha: vec![0.15, 0.55],
                    beta: vec![1.15],
                    kappa: 0.25,
                    omega_sq: 1.15,
                },
                MtpPosteriorDraw {
                    alpha: vec![0.25, 0.65],
                    beta: vec![1.25],
                    kappa: 0.35,
                    omega_sq: 1.25,
                },
                MtpPosteriorDraw {
                    alpha: vec![0.35, 0.75],
                    beta: vec![1.35],
                    kappa: 0.45,
                    omega_sq: 1.35,
                },
            ],
        };

        let summary =
            summarize_multi_chain_convergence(&[chain_a, chain_b]).expect("summary should work");
        assert_eq!(summary.chain_count, 2);
        assert_eq!(summary.draws_per_chain_used, 4);
        assert_eq!(summary.alpha_split_rhat.len(), 2);
        assert_eq!(summary.beta_split_rhat.len(), 1);
        assert!(summary.max_split_rhat.is_some());
    }
}
