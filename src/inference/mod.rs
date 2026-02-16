//! Reusable inference and MCMC utility types.

use thiserror::Error;

/// Errors for generic MCMC configuration.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum InferenceError {
    #[error("iterations must be positive")]
    InvalidIterations,
    #[error("burn-in ({burn_in}) must be smaller than iterations ({iterations})")]
    InvalidBurnIn { burn_in: usize, iterations: usize },
    #[error("thinning interval must be positive")]
    InvalidThinning,
}

/// Generic MCMC schedule.
#[derive(Debug, Clone, Copy)]
pub struct McmcConfig {
    pub iterations: usize,
    pub burn_in: usize,
    pub thin: usize,
    pub seed: u64,
    pub adapt_during_burn_in: bool,
}

impl Default for McmcConfig {
    fn default() -> Self {
        Self {
            iterations: 4_000,
            burn_in: 1_000,
            thin: 4,
            seed: 42,
            adapt_during_burn_in: true,
        }
    }
}

impl McmcConfig {
    /// # Errors
    ///
    /// Returns `InferenceError` if schedule values are invalid.
    pub const fn validate(self) -> Result<(), InferenceError> {
        if self.iterations == 0 {
            return Err(InferenceError::InvalidIterations);
        }
        if self.burn_in >= self.iterations {
            return Err(InferenceError::InvalidBurnIn {
                burn_in: self.burn_in,
                iterations: self.iterations,
            });
        }
        if self.thin == 0 {
            return Err(InferenceError::InvalidThinning);
        }
        Ok(())
    }

    /// Number of retained draws implied by this configuration.
    #[must_use]
    pub const fn retained_draws(self) -> usize {
        (self.iterations - self.burn_in) / self.thin
    }
}

/// Proposal counters for a single Metropolis-Hastings block.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProposalStats {
    pub proposed: usize,
    pub accepted: usize,
}

impl ProposalStats {
    /// Record one proposal and whether it was accepted.
    pub const fn record(&mut self, accepted: bool) {
        self.proposed += 1;
        if accepted {
            self.accepted += 1;
        }
    }

    /// Acceptance rate in `[0, 1]`, or `0` if no proposals were made.
    #[must_use]
    pub fn acceptance_rate(self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            usize_to_f64(self.accepted) / usize_to_f64(self.proposed)
        }
    }
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation_rejects_zero_iterations() {
        let config = McmcConfig {
            iterations: 0,
            ..McmcConfig::default()
        };
        assert_eq!(config.validate(), Err(InferenceError::InvalidIterations));
    }

    #[test]
    fn proposal_stats_tracks_acceptance() {
        let mut stats = ProposalStats::default();
        stats.record(true);
        stats.record(false);
        assert!((stats.acceptance_rate() - 0.5).abs() < 1.0e-12);
    }
}
