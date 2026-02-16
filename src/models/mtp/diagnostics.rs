//! MCMC and posterior predictive diagnostics for MTP.

use num_traits::ToPrimitive;

use crate::input::LongitudinalModelInput;

use super::effects::EffectIntervalSummary;
use super::likelihood::logistic_stable;
use super::posterior::MtpPosteriorSamples;
use super::types::{MtpConvergenceSummary, MtpError};

/// Lag-`k` autocorrelation for a scalar chain.
#[must_use]
pub fn autocorrelation(series: &[f64], lag: usize) -> f64 {
    if series.is_empty() || lag >= series.len() {
        return 0.0;
    }

    let n = series.len() - lag;
    let mean = series.iter().sum::<f64>() / usize_to_f64(series.len());

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for value in series {
        let centered = value - mean;
        denominator += centered * centered;
    }

    if denominator <= 0.0 {
        return 0.0;
    }

    for idx in 0..n {
        numerator += (series[idx] - mean) * (series[idx + lag] - mean);
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

    let mut rho_sum = 0.0;
    for lag in 1..n {
        let rho = autocorrelation(series, lag);
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
    if chains.len() < 2 {
        return Err(MtpError::InvalidChainCount {
            min: 2,
            found: chains.len(),
        });
    }

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
            if draw.alpha.len() != alpha_len || draw.beta.len() != beta_len {
                return Err(MtpError::InconsistentPosteriorDimensions);
            }
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
    let mut predicted_zero_rate_draws = Vec::with_capacity(samples.len());
    let mut predicted_mean_draws = Vec::with_capacity(samples.len());
    let mut brier_draws = Vec::with_capacity(samples.len());

    for draw in &samples.draws {
        if draw.alpha.len() != alpha_len {
            return Err(MtpError::DesignCoefficientMismatch {
                design_cols: alpha_len,
                coef_len: draw.alpha.len(),
            });
        }
        if draw.beta.len() != beta_len {
            return Err(MtpError::DesignCoefficientMismatch {
                design_cols: beta_len,
                coef_len: draw.beta.len(),
            });
        }

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
            brier_sum += (p_positive - observed_positive) * (p_positive - observed_positive);
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
        predicted_zero_rate: summarize_draws(&predicted_zero_rate_draws),
        predicted_mean: summarize_draws(&predicted_mean_draws),
        brier_score: summarize_draws(&brier_draws),
        calibration_bins: calibration_bins_summary(
            &predicted_probability_mean,
            input,
            calibration_bins,
        ),
    })
}

fn calibration_bins_summary(
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
    let mut split_chains = Vec::with_capacity(chains.len() * 2);

    for chain in chains {
        let first_half = chain
            .draws
            .iter()
            .take(half)
            .map(&extractor)
            .collect::<Vec<_>>();
        let second_half = chain
            .draws
            .iter()
            .skip(half)
            .take(half)
            .map(&extractor)
            .collect::<Vec<_>>();
        split_chains.push(first_half);
        split_chains.push(second_half);
    }

    split_rhat_scalar(&split_chains)
}

fn split_rhat_scalar(chains: &[Vec<f64>]) -> Result<f64, MtpError> {
    if chains.len() < 2 {
        return Err(MtpError::InvalidChainCount {
            min: 2,
            found: chains.len(),
        });
    }

    let n = chains.first().map_or(0, Vec::len);
    if n < 2 {
        return Err(MtpError::InsufficientChainDraws {
            minimum: 2,
            found: n,
        });
    }
    if chains.iter().any(|chain| chain.len() != n) {
        return Err(MtpError::InconsistentPosteriorDimensions);
    }

    let chain_means = chains
        .iter()
        .map(|chain| chain.iter().sum::<f64>() / usize_to_f64(n))
        .collect::<Vec<_>>();
    let chain_vars = chains
        .iter()
        .zip(chain_means.iter())
        .map(|(chain, mean)| sample_variance(chain, *mean))
        .collect::<Vec<_>>();

    let m = chains.len();
    let mean_of_means = chain_means.iter().sum::<f64>() / usize_to_f64(m);
    let between = usize_to_f64(n)
        * chain_means
            .iter()
            .map(|mean| {
                let centered = *mean - mean_of_means;
                centered * centered
            })
            .sum::<f64>()
        / usize_to_f64(m - 1);
    let within = chain_vars.iter().sum::<f64>() / usize_to_f64(m);

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

fn sample_variance(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / usize_to_f64(values.len() - 1)
}

fn summarize_draws(values: &[f64]) -> EffectIntervalSummary {
    if values.is_empty() {
        return EffectIntervalSummary::default();
    }

    let mean = values.iter().sum::<f64>() / usize_to_f64(values.len());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    EffectIntervalSummary {
        mean,
        q025: percentile(&sorted, 0.025),
        q50: percentile(&sorted, 0.5),
        q975: percentile(&sorted, 0.975),
    }
}

fn percentile(sorted_values: &[f64], probability: f64) -> f64 {
    if sorted_values.is_empty() {
        return f64::NAN;
    }

    let clamped = probability.clamp(0.0, 1.0);
    let last = sorted_values.len() - 1;
    let position = clamped * usize_to_f64(last);
    let lower = position.floor().to_usize().unwrap_or(0);
    let upper = position.ceil().to_usize().unwrap_or(last);

    if lower == upper {
        sorted_values[lower]
    } else {
        let weight = position - usize_to_f64(lower);
        (1.0 - weight).mul_add(sorted_values[lower], weight * sorted_values[upper])
    }
}

fn dot_row(matrix: &Mat<f64>, row: usize, coef: &[f64]) -> f64 {
    (0..matrix.ncols())
        .map(|col| matrix[(row, col)] * coef[col])
        .sum()
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

use faer::Mat;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::mtp::posterior::MtpPosteriorDraw;

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
