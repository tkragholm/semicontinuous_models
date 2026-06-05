//! Posterior storage and summaries for MTP.

use crate::utils::{calculate_quantile, usize_to_f64};

/// A single posterior draw from the MTP parameter space.
#[derive(Debug, Clone)]
pub struct MtpPosteriorDraw {
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    pub kappa: f64,
    pub omega_sq: f64,
}

/// Posterior draw collection.
#[derive(Debug, Clone, Default)]
pub struct MtpPosteriorSamples {
    pub draws: Vec<MtpPosteriorDraw>,
}

impl MtpPosteriorSamples {
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.draws.len()
    }
}

/// Scalar posterior summary statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParameterSummary {
    pub mean: f64,
    pub std_dev: f64,
    pub q025: f64,
    pub q50: f64,
    pub q975: f64,
}

/// Posterior summary for the MTP parameter blocks.
#[derive(Debug, Clone, Default)]
pub struct MtpPosteriorSummary {
    pub alpha: Vec<ParameterSummary>,
    pub beta: Vec<ParameterSummary>,
    pub kappa: Option<ParameterSummary>,
    pub omega_sq: Option<ParameterSummary>,
    pub draw_count: usize,
}

/// Compute posterior summaries for all stored parameter blocks.
#[must_use]
pub fn summarize_posterior(samples: &MtpPosteriorSamples) -> MtpPosteriorSummary {
    let draw_count = samples.len();
    if draw_count == 0 {
        return MtpPosteriorSummary {
            draw_count,
            ..MtpPosteriorSummary::default()
        };
    }

    let alpha_len = samples.draws.first().map_or(0, |draw| draw.alpha.len());
    let beta_len = samples.draws.first().map_or(0, |draw| draw.beta.len());

    // One scratch buffer reused across every coefficient and block, instead of
    // allocating a fresh `Vec` (twice) per coefficient.
    let mut buffer: Vec<f64> = Vec::with_capacity(draw_count);

    let mut alpha = Vec::with_capacity(alpha_len);
    for index in 0..alpha_len {
        buffer.clear();
        buffer.extend(samples.draws.iter().map(|draw| draw.alpha[index]));
        alpha.push(summarize_scalar_in_place(&mut buffer));
    }

    let mut beta = Vec::with_capacity(beta_len);
    for index in 0..beta_len {
        buffer.clear();
        buffer.extend(samples.draws.iter().map(|draw| draw.beta[index]));
        beta.push(summarize_scalar_in_place(&mut buffer));
    }

    buffer.clear();
    buffer.extend(samples.draws.iter().map(|draw| draw.kappa));
    let kappa = Some(summarize_scalar_in_place(&mut buffer));

    buffer.clear();
    buffer.extend(samples.draws.iter().map(|draw| draw.omega_sq));
    let omega_sq = Some(summarize_scalar_in_place(&mut buffer));

    MtpPosteriorSummary {
        alpha,
        beta,
        kappa,
        omega_sq,
        draw_count,
    }
}

/// Summarize a scalar parameter from a caller-owned buffer: mean and variance
/// in one pass, then sort `values` in place for the quantiles (no internal
/// `to_vec`).
#[must_use]
fn summarize_scalar_in_place(values: &mut [f64]) -> ParameterSummary {
    if values.is_empty() {
        return ParameterSummary::default();
    }

    let n = usize_to_f64(values.len());
    let mean = values.iter().sum::<f64>() / n;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / n.max(1.0);

    values.sort_by(f64::total_cmp);

    ParameterSummary {
        mean,
        std_dev: variance.sqrt(),
        q025: calculate_quantile(values, 0.025),
        q50: calculate_quantile(values, 0.5),
        q975: calculate_quantile(values, 0.975),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_empty_samples() {
        let summary = summarize_posterior(&MtpPosteriorSamples::default());
        assert_eq!(summary.draw_count, 0);
        assert!(summary.alpha.is_empty());
        assert!(summary.beta.is_empty());
        assert!(summary.kappa.is_none());
    }

    #[test]
    fn summarize_non_empty_samples() {
        let samples = MtpPosteriorSamples {
            draws: vec![
                MtpPosteriorDraw {
                    alpha: vec![0.0],
                    beta: vec![1.0],
                    kappa: 0.1,
                    omega_sq: 0.8,
                },
                MtpPosteriorDraw {
                    alpha: vec![2.0],
                    beta: vec![3.0],
                    kappa: 0.3,
                    omega_sq: 1.2,
                },
            ],
        };

        let summary = summarize_posterior(&samples);
        assert_eq!(summary.draw_count, 2);
        assert_eq!(summary.alpha.len(), 1);
        assert!((summary.alpha[0].mean - 1.0).abs() < 1.0e-12);
    }
}
