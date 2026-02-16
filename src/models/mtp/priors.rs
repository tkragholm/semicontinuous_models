//! Prior specifications and log-density helpers for MTP.

use statrs::function::gamma::ln_gamma;

/// Hyperparameters for the correlated MTP prior set.
#[derive(Debug, Clone, Copy)]
pub struct MtpPriorConfig {
    /// Variance for Normal(0, variance) prior on binary coefficients.
    pub alpha_variance: f64,
    /// Variance for Normal(0, variance) prior on mean coefficients.
    pub beta_variance: f64,
    /// Lower bound for uniform prior on skewness parameter kappa.
    pub kappa_lower: f64,
    /// Upper bound for uniform prior on skewness parameter kappa.
    pub kappa_upper: f64,
    /// Shape parameter for inverse-gamma prior on omega squared.
    pub omega_sq_shape: f64,
    /// Scale parameter for inverse-gamma prior on omega squared.
    pub omega_sq_scale: f64,
    /// Degrees of freedom for inverse-Wishart random-effects covariance prior.
    pub random_effects_df: f64,
    /// Diagonal scale entry for inverse-Wishart covariance prior scale matrix.
    pub random_effects_scale_diag: f64,
    /// Variance for Normal(0, variance) prior on family-level binary intercept effects.
    pub family_binary_variance: f64,
    /// Variance for Normal(0, variance) prior on family-level mean intercept effects.
    pub family_mean_variance: f64,
}

impl Default for MtpPriorConfig {
    fn default() -> Self {
        Self {
            alpha_variance: 1_000.0,
            beta_variance: 1_000.0,
            kappa_lower: -10.0,
            kappa_upper: 10.0,
            omega_sq_shape: 0.001,
            omega_sq_scale: 0.001,
            random_effects_df: 4.0,
            random_effects_scale_diag: 1.0,
            family_binary_variance: 1.0,
            family_mean_variance: 1.0,
        }
    }
}

impl MtpPriorConfig {
    /// Whether all prior hyperparameters are numerically valid.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.alpha_variance > 0.0
            && self.beta_variance > 0.0
            && self.kappa_lower < self.kappa_upper
            && self.omega_sq_shape > 0.0
            && self.omega_sq_scale > 0.0
            && self.random_effects_df > 0.0
            && self.random_effects_scale_diag > 0.0
            && self.family_binary_variance > 0.0
            && self.family_mean_variance > 0.0
    }
}

/// Log-density for `Uniform(low, high)`.
#[must_use]
pub fn log_uniform_density(value: f64, low: f64, high: f64) -> f64 {
    if low.partial_cmp(&high) != Some(std::cmp::Ordering::Less) {
        return f64::NAN;
    }
    if (low..=high).contains(&value) {
        -(high - low).ln()
    } else {
        f64::NEG_INFINITY
    }
}

/// Log-density for an inverse-gamma distribution.
#[must_use]
pub fn log_inverse_gamma_density(value: f64, shape: f64, scale: f64) -> f64 {
    if !(value > 0.0 && shape > 0.0 && scale > 0.0) {
        return f64::NEG_INFINITY;
    }

    shape.mul_add(scale.ln(), -ln_gamma(shape)) - (shape + 1.0).mul_add(value.ln(), scale / value)
}

/// Log-density for `Normal(0, variance)`.
#[must_use]
pub fn log_zero_mean_normal_density(value: f64, variance: f64) -> f64 {
    if variance <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -0.5 * (std::f64::consts::TAU.ln() + variance.ln() + value * value / variance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_defaults_are_valid() {
        assert!(MtpPriorConfig::default().is_valid());
    }

    #[test]
    fn inverse_gamma_density_requires_positive_inputs() {
        let ll = log_inverse_gamma_density(0.0, 1.0, 1.0);
        assert!(!ll.is_finite());
    }
}
