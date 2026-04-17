//! Likelihood and numerical-stability helpers for MTP.

use statrs::function::erf::erf;

const EPS_PROBABILITY: f64 = 1.0e-12;

/// Stable logistic transform.
#[must_use]
pub fn logistic_stable(value: f64) -> f64 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

/// Bound probability away from exact 0 and 1.
#[must_use]
pub fn clamp_probability(probability: f64) -> f64 {
    probability.clamp(EPS_PROBABILITY, 1.0 - EPS_PROBABILITY)
}

/// Stable `log(1 - p)` with clipping.
#[must_use]
pub fn log_one_minus_probability(probability: f64) -> f64 {
    let p = clamp_probability(probability);
    (-p).ln_1p()
}

/// Log-density of standard normal.
#[must_use]
pub fn log_standard_normal_pdf(value: f64) -> f64 {
    -0.5 * value.mul_add(value, std::f64::consts::TAU.ln())
}

/// Log-CDF of standard normal with finite clipping.
#[must_use]
pub fn log_standard_normal_cdf(value: f64) -> f64 {
    let cdf = 0.5 * (1.0 + erf(value / std::f64::consts::SQRT_2));
    clamp_probability(cdf).ln()
}

/// Zero branch log-likelihood for `y == 0`.
#[must_use]
pub fn zero_branch_log_likelihood(prob_positive: f64) -> f64 {
    log_one_minus_probability(prob_positive)
}

/// Positive branch log-likelihood for `y > 0`.
#[must_use]
pub fn positive_branch_log_likelihood(
    outcome: f64,
    xi: f64,
    omega: f64,
    kappa: f64,
    prob_positive: f64,
) -> f64 {
    if !(outcome > 0.0 && omega > 0.0 && outcome.is_finite() && omega.is_finite()) {
        return f64::NEG_INFINITY;
    }

    let p = clamp_probability(prob_positive);
    let z = (outcome.ln() - xi) / omega;

    p.ln() + std::f64::consts::LN_2 - outcome.ln() - omega.ln()
        + log_standard_normal_pdf(z)
        + log_standard_normal_cdf(kappa * z)
}

/// Positive-branch log-likelihood for `y > 0` under a log-normal positive part.
#[must_use]
pub fn positive_branch_log_likelihood_lognormal(
    outcome: f64,
    xi: f64,
    omega: f64,
    prob_positive: f64,
) -> f64 {
    if !(outcome > 0.0 && omega > 0.0 && outcome.is_finite() && omega.is_finite()) {
        return f64::NEG_INFINITY;
    }

    let p = clamp_probability(prob_positive);
    let z = (outcome.ln() - xi) / omega;
    p.ln() - outcome.ln() - omega.ln() + log_standard_normal_pdf(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logistic_is_bounded() {
        let low = logistic_stable(-1_000.0);
        let high = logistic_stable(1_000.0);
        assert!(low >= 0.0);
        assert!(high <= 1.0);
    }

    #[test]
    fn positive_branch_rejects_non_positive_outcome() {
        let ll = positive_branch_log_likelihood(0.0, 0.0, 1.0, 0.0, 0.5);
        assert!(!ll.is_finite());
    }

    #[test]
    fn zero_branch_is_finite_interior() {
        let ll = zero_branch_log_likelihood(0.4);
        assert!(ll.is_finite());
    }

    #[test]
    fn skew_normal_reduces_to_lognormal_at_zero_kappa() {
        let outcome = 3.2;
        let xi = 1.1;
        let omega = 0.7;
        let p = 0.63;
        let skew = positive_branch_log_likelihood(outcome, xi, omega, 0.0, p);
        let lognormal = positive_branch_log_likelihood_lognormal(outcome, xi, omega, p);
        assert!((skew - lognormal).abs() < 1.0e-10);
    }
}
