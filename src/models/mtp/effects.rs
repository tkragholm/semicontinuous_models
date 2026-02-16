//! Counterfactual effect helpers derived from MTP posterior draws.

use faer::Mat;
use num_traits::ToPrimitive;

use super::likelihood::logistic_stable;
use super::posterior::MtpPosteriorSamples;
use super::types::MtpError;

/// Exposed/unexposed design scenarios for posterior contrast calculations.
#[derive(Debug, Clone)]
pub struct CounterfactualScenario {
    pub binary_design_exposed: Mat<f64>,
    pub mean_design_exposed: Mat<f64>,
    pub binary_design_unexposed: Mat<f64>,
    pub mean_design_unexposed: Mat<f64>,
}

/// Per-period effect estimates.
#[derive(Debug, Clone, Copy)]
pub struct PeriodEffect {
    pub period_index: usize,
    pub mean_exposed: f64,
    pub mean_unexposed: f64,
    pub additive_effect: f64,
    pub multiplicative_effect: f64,
    pub odds_ratio_positive: f64,
}

/// Counterfactual effects across all periods in a scenario.
#[derive(Debug, Clone, Default)]
pub struct CounterfactualEffects {
    pub per_period: Vec<PeriodEffect>,
    pub cumulative_additive_effect: f64,
}

/// Posterior summary interval for a scalar effect quantity.
#[derive(Debug, Clone, Copy, Default)]
pub struct EffectIntervalSummary {
    pub mean: f64,
    pub q025: f64,
    pub q50: f64,
    pub q975: f64,
}

/// Posterior summaries for period-specific effects.
#[derive(Debug, Clone)]
pub struct PeriodEffectSummary {
    pub period_index: usize,
    pub mean_exposed: EffectIntervalSummary,
    pub mean_unexposed: EffectIntervalSummary,
    pub additive_effect: EffectIntervalSummary,
    pub multiplicative_effect: EffectIntervalSummary,
    pub odds_ratio_positive: EffectIntervalSummary,
}

/// Posterior summaries for all counterfactual effects.
#[derive(Debug, Clone, Default)]
pub struct CounterfactualEffectsSummary {
    pub per_period: Vec<PeriodEffectSummary>,
    pub cumulative_additive_effect: EffectIntervalSummary,
}

/// # Errors
///
/// Returns `MtpError` if draws are missing or design dimensions are inconsistent.
pub fn compute_counterfactual_effects(
    samples: &MtpPosteriorSamples,
    scenario: &CounterfactualScenario,
) -> Result<CounterfactualEffects, MtpError> {
    if samples.is_empty() {
        return Err(MtpError::EmptyPosterior);
    }

    validate_counterfactual_design(scenario)?;

    let alpha = posterior_mean_alpha(samples);
    let beta = posterior_mean_beta(samples);

    validate_coef_dims(scenario.binary_design_exposed.ncols(), alpha.len())?;
    validate_coef_dims(scenario.binary_design_unexposed.ncols(), alpha.len())?;
    validate_coef_dims(scenario.mean_design_exposed.ncols(), beta.len())?;
    validate_coef_dims(scenario.mean_design_unexposed.ncols(), beta.len())?;

    let periods = scenario.binary_design_exposed.nrows();
    let mut per_period = Vec::with_capacity(periods);
    let mut cumulative_additive = 0.0;

    for row in 0..periods {
        let p_exposed = logistic_stable(dot_row(&scenario.binary_design_exposed, row, &alpha));
        let p_unexposed = logistic_stable(dot_row(&scenario.binary_design_unexposed, row, &alpha));

        // MTP beta parameterizes the marginal mean directly on log-scale.
        let mean_exposed = dot_row(&scenario.mean_design_exposed, row, &beta).exp();
        let mean_unexposed = dot_row(&scenario.mean_design_unexposed, row, &beta).exp();

        let additive_effect = mean_exposed - mean_unexposed;
        let multiplicative_effect = mean_exposed / mean_unexposed.max(1.0e-12);
        let odds_exposed = p_exposed / (1.0 - p_exposed).max(1.0e-12);
        let odds_unexposed = p_unexposed / (1.0 - p_unexposed).max(1.0e-12);

        cumulative_additive += additive_effect;
        per_period.push(PeriodEffect {
            period_index: row,
            mean_exposed,
            mean_unexposed,
            additive_effect,
            multiplicative_effect,
            odds_ratio_positive: odds_exposed / odds_unexposed.max(1.0e-12),
        });
    }

    Ok(CounterfactualEffects {
        per_period,
        cumulative_additive_effect: cumulative_additive,
    })
}

/// Compute posterior effect summaries across all retained draws.
///
/// # Errors
///
/// Returns `MtpError` if draws are missing or design dimensions are inconsistent.
pub fn compute_counterfactual_effects_summary(
    samples: &MtpPosteriorSamples,
    scenario: &CounterfactualScenario,
) -> Result<CounterfactualEffectsSummary, MtpError> {
    if samples.is_empty() {
        return Err(MtpError::EmptyPosterior);
    }
    validate_counterfactual_design(scenario)?;

    let alpha_len = samples.draws.first().map_or(0, |draw| draw.alpha.len());
    let beta_len = samples.draws.first().map_or(0, |draw| draw.beta.len());
    validate_coef_dims(scenario.binary_design_exposed.ncols(), alpha_len)?;
    validate_coef_dims(scenario.binary_design_unexposed.ncols(), alpha_len)?;
    validate_coef_dims(scenario.mean_design_exposed.ncols(), beta_len)?;
    validate_coef_dims(scenario.mean_design_unexposed.ncols(), beta_len)?;

    let periods = scenario.binary_design_exposed.nrows();
    let draw_count = samples.len();

    let mut mean_exposed_draws = vec![Vec::with_capacity(draw_count); periods];
    let mut mean_unexposed_draws = vec![Vec::with_capacity(draw_count); periods];
    let mut additive_draws = vec![Vec::with_capacity(draw_count); periods];
    let mut multiplicative_draws = vec![Vec::with_capacity(draw_count); periods];
    let mut odds_ratio_draws = vec![Vec::with_capacity(draw_count); periods];
    let mut cumulative_additive_draws = Vec::with_capacity(draw_count);

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

        let mut cumulative_additive = 0.0;
        for row in 0..periods {
            let p_exposed =
                logistic_stable(dot_row(&scenario.binary_design_exposed, row, &draw.alpha));
            let p_unexposed =
                logistic_stable(dot_row(&scenario.binary_design_unexposed, row, &draw.alpha));

            let mean_exposed = dot_row(&scenario.mean_design_exposed, row, &draw.beta).exp();
            let mean_unexposed = dot_row(&scenario.mean_design_unexposed, row, &draw.beta).exp();
            let additive_effect = mean_exposed - mean_unexposed;
            let multiplicative_effect = mean_exposed / mean_unexposed.max(1.0e-12);
            let odds_exposed = p_exposed / (1.0 - p_exposed).max(1.0e-12);
            let odds_unexposed = p_unexposed / (1.0 - p_unexposed).max(1.0e-12);
            let odds_ratio_positive = odds_exposed / odds_unexposed.max(1.0e-12);

            mean_exposed_draws[row].push(mean_exposed);
            mean_unexposed_draws[row].push(mean_unexposed);
            additive_draws[row].push(additive_effect);
            multiplicative_draws[row].push(multiplicative_effect);
            odds_ratio_draws[row].push(odds_ratio_positive);
            cumulative_additive += additive_effect;
        }
        cumulative_additive_draws.push(cumulative_additive);
    }

    let per_period = (0..periods)
        .map(|row| PeriodEffectSummary {
            period_index: row,
            mean_exposed: summarize_draws(&mean_exposed_draws[row]),
            mean_unexposed: summarize_draws(&mean_unexposed_draws[row]),
            additive_effect: summarize_draws(&additive_draws[row]),
            multiplicative_effect: summarize_draws(&multiplicative_draws[row]),
            odds_ratio_positive: summarize_draws(&odds_ratio_draws[row]),
        })
        .collect();

    Ok(CounterfactualEffectsSummary {
        per_period,
        cumulative_additive_effect: summarize_draws(&cumulative_additive_draws),
    })
}

fn validate_counterfactual_design(scenario: &CounterfactualScenario) -> Result<(), MtpError> {
    let rows = scenario.binary_design_exposed.nrows();
    if rows == 0
        || scenario.mean_design_exposed.nrows() == 0
        || scenario.binary_design_unexposed.nrows() == 0
        || scenario.mean_design_unexposed.nrows() == 0
    {
        return Err(MtpError::EmptyCounterfactualDesign);
    }

    if scenario.mean_design_exposed.nrows() != rows
        || scenario.binary_design_unexposed.nrows() != rows
        || scenario.mean_design_unexposed.nrows() != rows
    {
        return Err(MtpError::CounterfactualRowMismatch);
    }

    Ok(())
}

const fn validate_coef_dims(design_cols: usize, coef_len: usize) -> Result<(), MtpError> {
    if design_cols != coef_len {
        return Err(MtpError::DesignCoefficientMismatch {
            design_cols,
            coef_len,
        });
    }
    Ok(())
}

fn posterior_mean_alpha(samples: &MtpPosteriorSamples) -> Vec<f64> {
    let len = samples.draws.first().map_or(0, |draw| draw.alpha.len());
    let mut mean = vec![0.0; len];

    for draw in &samples.draws {
        for (index, value) in draw.alpha.iter().copied().enumerate() {
            mean[index] += value;
        }
    }

    let n = usize_to_f64(samples.len());
    for value in &mut mean {
        *value /= n;
    }

    mean
}

fn posterior_mean_beta(samples: &MtpPosteriorSamples) -> Vec<f64> {
    let len = samples.draws.first().map_or(0, |draw| draw.beta.len());
    let mut mean = vec![0.0; len];

    for draw in &samples.draws {
        for (index, value) in draw.beta.iter().copied().enumerate() {
            mean[index] += value;
        }
    }

    let n = usize_to_f64(samples.len());
    for value in &mut mean {
        *value /= n;
    }

    mean
}

fn dot_row(matrix: &Mat<f64>, row: usize, coef: &[f64]) -> f64 {
    (0..matrix.ncols())
        .map(|col| matrix[(row, col)] * coef[col])
        .sum()
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

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::mtp::posterior::MtpPosteriorDraw;

    #[test]
    fn compute_effects_requires_draws() {
        let scenario = CounterfactualScenario {
            binary_design_exposed: Mat::from_fn(1, 1, |_i, _j| 1.0),
            mean_design_exposed: Mat::from_fn(1, 1, |_i, _j| 1.0),
            binary_design_unexposed: Mat::from_fn(1, 1, |_i, _j| 1.0),
            mean_design_unexposed: Mat::from_fn(1, 1, |_i, _j| 1.0),
        };

        let result = compute_counterfactual_effects(&MtpPosteriorSamples::default(), &scenario);
        assert!(matches!(result, Err(MtpError::EmptyPosterior)));
    }

    #[test]
    fn compute_effects_runs_for_matching_dims() {
        let samples = MtpPosteriorSamples {
            draws: vec![MtpPosteriorDraw {
                alpha: vec![0.0, 0.5],
                beta: vec![0.0, 0.25],
                kappa: 0.1,
                omega_sq: 1.0,
            }],
        };

        let scenario = CounterfactualScenario {
            binary_design_exposed: Mat::from_fn(2, 2, |i, j| {
                if j == 0 {
                    1.0
                } else if i == 0 {
                    0.0
                } else {
                    1.0
                }
            }),
            mean_design_exposed: Mat::from_fn(2, 2, |i, j| {
                if j == 0 {
                    1.0
                } else if i == 0 {
                    0.0
                } else {
                    1.0
                }
            }),
            binary_design_unexposed: Mat::from_fn(2, 2, |_i, j| if j == 0 { 1.0 } else { 0.0 }),
            mean_design_unexposed: Mat::from_fn(2, 2, |_i, j| if j == 0 { 1.0 } else { 0.0 }),
        };

        let effects = compute_counterfactual_effects(&samples, &scenario)
            .expect("matching dimensions should produce effects");
        assert_eq!(effects.per_period.len(), 2);
    }

    #[test]
    fn compute_effects_summary_requires_draws() {
        let scenario = CounterfactualScenario {
            binary_design_exposed: Mat::from_fn(1, 1, |_i, _j| 1.0),
            mean_design_exposed: Mat::from_fn(1, 1, |_i, _j| 1.0),
            binary_design_unexposed: Mat::from_fn(1, 1, |_i, _j| 1.0),
            mean_design_unexposed: Mat::from_fn(1, 1, |_i, _j| 1.0),
        };

        let result =
            compute_counterfactual_effects_summary(&MtpPosteriorSamples::default(), &scenario);
        assert!(matches!(result, Err(MtpError::EmptyPosterior)));
    }

    #[test]
    fn compute_effects_summary_returns_intervals() {
        let samples = MtpPosteriorSamples {
            draws: vec![
                MtpPosteriorDraw {
                    alpha: vec![0.0, 0.5],
                    beta: vec![0.0, 0.25],
                    kappa: 0.1,
                    omega_sq: 1.0,
                },
                MtpPosteriorDraw {
                    alpha: vec![0.0, 0.4],
                    beta: vec![0.0, 0.2],
                    kappa: 0.2,
                    omega_sq: 1.1,
                },
            ],
        };

        let scenario = CounterfactualScenario {
            binary_design_exposed: Mat::from_fn(2, 2, |i, j| {
                if j == 0 {
                    1.0
                } else if i == 0 {
                    0.0
                } else {
                    1.0
                }
            }),
            mean_design_exposed: Mat::from_fn(2, 2, |i, j| {
                if j == 0 {
                    1.0
                } else if i == 0 {
                    0.0
                } else {
                    1.0
                }
            }),
            binary_design_unexposed: Mat::from_fn(2, 2, |_i, j| if j == 0 { 1.0 } else { 0.0 }),
            mean_design_unexposed: Mat::from_fn(2, 2, |_i, j| if j == 0 { 1.0 } else { 0.0 }),
        };

        let summary = compute_counterfactual_effects_summary(&samples, &scenario)
            .expect("summary should compute");
        assert_eq!(summary.per_period.len(), 2);
        assert!(summary.cumulative_additive_effect.mean.is_finite());
        assert!(
            summary.per_period[0].additive_effect.q975
                >= summary.per_period[0].additive_effect.q025
        );
    }
}
