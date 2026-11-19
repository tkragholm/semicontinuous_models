//! Core public types for the MTP module.

use super::posterior::MtpPosteriorSummary;
use super::priors::MtpPriorConfig;
use crate::input::LongitudinalInputError;
use crate::models::{FitMetadata, Model};
use faer::Mat;
use thiserror::Error;

/// Errors returned by MTP configuration, validation, and fitting.
#[derive(Debug, Error)]
pub enum MtpError {
    #[error(transparent)]
    InvalidInput(#[from] LongitudinalInputError),
    #[error("iterations must be positive")]
    InvalidIterations,
    #[error("burn-in ({burn_in}) must be smaller than iterations ({iterations})")]
    InvalidBurnIn { burn_in: usize, iterations: usize },
    #[error("thinning interval must be positive")]
    InvalidThinning,
    #[error("design columns ({design_cols}) must match coefficient length ({coef_len})")]
    DesignCoefficientMismatch { design_cols: usize, coef_len: usize },
    #[error("counterfactual design row counts must match")]
    CounterfactualRowMismatch,
    #[error("counterfactual design matrices must be non-empty")]
    EmptyCounterfactualDesign,
    #[error("posterior draws are required")]
    EmptyPosterior,
    #[error("calibration bin count must be positive")]
    InvalidCalibrationBins,
    #[error("multi-chain workflows require at least {min} chains; found {found}")]
    InvalidChainCount { min: usize, found: usize },
    #[error("each chain must retain at least {minimum} draws; minimum found {found}")]
    InsufficientChainDraws { minimum: usize, found: usize },
    #[error("posterior dimensions differ across chains")]
    InconsistentPosteriorDimensions,
    #[error("invalid MTP prior configuration")]
    InvalidPriorConfig,
    #[error("invalid MTP proposal tuning configuration")]
    InvalidProposalTuning,
    #[error("family random-effects layer requires family ids in longitudinal input")]
    MissingFamilyIds,
    #[error("MTP fitting requires at least one positive outcome")]
    NoPositiveOutcomes,
    #[error("MTP solver failed")]
    SolveFailed,
    #[error("invalid weights in a weighted product: {0}")]
    InvalidWeights(#[from] epistat_linalg::DesignError),
    #[error("MTP fitting did not converge")]
    NonConvergence,
    #[error("MTP sampler not implemented yet: {0}")]
    NotImplemented(&'static str),
}

/// # Errors
///
/// Returns `MtpError::InvalidChainCount` if `found < min`.
pub const fn validate_chain_count(found: usize, min: usize) -> Result<(), MtpError> {
    if found < min {
        return Err(MtpError::InvalidChainCount { min, found });
    }
    Ok(())
}

/// Random-effects structure used by the longitudinal MTP model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomEffectsStructure {
    /// Subject-level random intercepts for binary and mean parts.
    InterceptsOnly,
    /// Subject-level random intercepts and linear time slopes for both parts.
    InterceptsAndTimeSlopes,
}

impl RandomEffectsStructure {
    /// Number of random-effect dimensions per subject implied by the structure.
    #[must_use]
    pub const fn random_effect_dimension(self) -> usize {
        match self {
            Self::InterceptsOnly => 2,
            Self::InterceptsAndTimeSlopes => 4,
        }
    }
}

/// Positive-part distribution for outcomes where `y > 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PositivePartDistribution {
    /// Log-skew-normal positive part with skewness parameter `kappa`.
    #[default]
    LogSkewNormal,
    /// Log-normal positive part (equivalent to skew-normal with fixed `kappa = 0`).
    LogNormal,
}

/// Optional family-level random-effects layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FamilyRandomEffects {
    /// No family-level random effects.
    #[default]
    Disabled,
    /// Family-level random intercepts for binary and mean components.
    InterceptsOnly,
}

/// Sampler configuration for MTP fitting.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct MtpFitOptions {
    /// Total MCMC iterations.
    #[builder(default = 4_000_usize)]
    pub iterations: usize,
    /// Burn-in iterations discarded before retention.
    #[builder(default = 1_000_usize)]
    pub burn_in: usize,
    /// Keep every `thin`-th draw after burn-in.
    #[builder(default = 4_usize)]
    pub thin: usize,
    /// RNG seed for reproducibility.
    #[builder(default = 42_u64)]
    pub seed: u64,
    /// Random-effects structure.
    #[builder(default = RandomEffectsStructure::InterceptsOnly)]
    pub random_effects: RandomEffectsStructure,
    /// Enable adaptation during burn-in.
    #[builder(default = true)]
    pub adapt_during_burn_in: bool,
}

impl Default for MtpFitOptions {
    fn default() -> Self {
        Self {
            iterations: 4_000,
            burn_in: 1_000,
            thin: 4,
            seed: 42,
            random_effects: RandomEffectsStructure::InterceptsOnly,
            adapt_during_burn_in: true,
        }
    }
}

impl MtpFitOptions {
    /// # Errors
    ///
    /// Returns `MtpError` if options are internally inconsistent.
    pub const fn validate(self) -> Result<(), MtpError> {
        if self.iterations == 0 {
            return Err(MtpError::InvalidIterations);
        }
        if self.burn_in >= self.iterations {
            return Err(MtpError::InvalidBurnIn {
                burn_in: self.burn_in,
                iterations: self.iterations,
            });
        }
        if self.thin == 0 {
            return Err(MtpError::InvalidThinning);
        }
        Ok(())
    }

    /// Number of retained draws implied by `(iterations, burn_in, thin)`.
    #[must_use]
    pub const fn retained_draws(self) -> usize {
        crate::utils::retained_draws(self.iterations, self.burn_in, self.thin)
    }
}

/// Validate the posterior draw dimensions used by MTP summary calculations.
pub(super) const fn validate_draw_dimensions(
    alpha_len: usize,
    beta_len: usize,
    draw_alpha_len: usize,
    draw_beta_len: usize,
) -> Result<(), MtpError> {
    if draw_alpha_len != alpha_len {
        return Err(MtpError::DesignCoefficientMismatch {
            design_cols: alpha_len,
            coef_len: draw_alpha_len,
        });
    }
    if draw_beta_len != beta_len {
        return Err(MtpError::DesignCoefficientMismatch {
            design_cols: beta_len,
            coef_len: draw_beta_len,
        });
    }
    Ok(())
}

/// Proposal-scale and adaptation controls for MTP MCMC.
///
/// `alpha`, `beta` and `log(omega_sq)` are updated as one block whose proposal
/// covariance is estimated from the chain during burn-in. `kappa` has its own random
/// walk, which holds the variance of the log positive part fixed. A subject's
/// random-effect step is a multiple of the Cholesky factor of the current random-effects
/// covariance.
///
/// Every `adaptation_interval` burn-in iterations each step size `s` takes a
/// Robbins-Monro step on the log scale towards the middle of the acceptance band,
/// `log(s) += adaptation_gain / sqrt(k) * (rate - target)`, where `rate` is its
/// acceptance rate over the interval and `k` counts its adaptations (Andrieu and Thoms
/// 2008). The step is not capped, so the gains sum without bound and any starting scale
/// can be corrected given enough intervals. All adaptation stops at the end of burn-in.
///
/// Andrieu, C. and Thoms, J. (2008). A tutorial on adaptive MCMC. Statistics and
/// Computing 18(4), 343-373.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct MtpProposalTuning {
    /// Minimum allowed proposal scale.
    #[builder(default = 1.0e-3_f64)]
    pub min_draw_scale: f64,
    /// Initial random-walk scale for `kappa`.
    #[builder(default = 0.05_f64)]
    pub kappa_draw_scale: f64,
    /// Initial proposal standard deviation of `log(omega_sq)` in the fixed-effect block,
    /// used until the block covariance is estimated from the chain.
    #[builder(default = 0.03_f64)]
    pub log_omega_draw_scale: f64,
    /// Adapt every `adaptation_interval` iterations during burn-in.
    #[builder(default = 50_usize)]
    pub adaptation_interval: usize,
    /// Lower end of the acceptance band; adaptation targets the band's midpoint.
    #[builder(default = 0.2_f64)]
    pub acceptance_target_low: f64,
    /// Upper end of the acceptance band; adaptation targets the band's midpoint.
    #[builder(default = 0.35_f64)]
    pub acceptance_target_high: f64,
    /// Gain `c` of the Robbins-Monro step `c / sqrt(k)` on the log step size.
    #[builder(default = 3.0_f64)]
    pub adaptation_gain: f64,
    /// Initial random-walk scale for family-level random effects.
    #[builder(default = 0.05_f64)]
    pub family_effect_draw_scale: f64,
}

impl Default for MtpProposalTuning {
    fn default() -> Self {
        Self {
            min_draw_scale: 1.0e-3,
            kappa_draw_scale: 0.05,
            log_omega_draw_scale: 0.03,
            adaptation_interval: 50,
            acceptance_target_low: 0.2,
            acceptance_target_high: 0.35,
            adaptation_gain: 3.0,
            family_effect_draw_scale: 0.05,
        }
    }
}

impl MtpProposalTuning {
    /// Whether proposal tuning settings are numerically valid.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.min_draw_scale > 0.0
            && self.kappa_draw_scale > 0.0
            && self.log_omega_draw_scale > 0.0
            && self.adaptation_interval > 0
            && self.acceptance_target_low >= 0.0
            && self.acceptance_target_high <= 1.0
            && self.acceptance_target_low < self.acceptance_target_high
            && self.adaptation_gain > 0.0
            && self.family_effect_draw_scale > 0.0
    }
}

/// Full sampler configuration for MTP fitting.
#[derive(bon::Builder, Debug, Clone, Copy, Default)]
pub struct MtpSamplerConfig {
    #[builder(default)]
    pub fit_options: MtpFitOptions,
    #[builder(default)]
    pub prior_config: MtpPriorConfig,
    #[builder(default)]
    pub proposal_tuning: MtpProposalTuning,
    #[builder(default)]
    pub positive_part_distribution: PositivePartDistribution,
    #[builder(default)]
    pub family_random_effects: FamilyRandomEffects,
}

impl MtpSamplerConfig {
    /// # Errors
    ///
    /// Returns `MtpError` if any configuration block is invalid.
    pub fn validate(self) -> Result<(), MtpError> {
        self.fit_options.validate()?;
        if !self.prior_config.is_valid() {
            return Err(MtpError::InvalidPriorConfig);
        }
        if !self.proposal_tuning.is_valid() {
            return Err(MtpError::InvalidProposalTuning);
        }
        Ok(())
    }
}

/// Configuration for running multiple independent MCMC chains.
#[derive(Debug, Clone, Copy)]
pub struct MtpMultiChainOptions {
    /// Number of independent chains to run.
    ///
    /// Chain `i` draws from `epistat_rng::replicate_seed(seed, i)`, so a chain
    /// does not depend on the other chains or on the thread that runs it.
    pub chains: usize,
}

impl Default for MtpMultiChainOptions {
    fn default() -> Self {
        Self { chains: 4 }
    }
}

impl MtpMultiChainOptions {
    /// # Errors
    ///
    /// Returns `MtpError` if multi-chain options are invalid.
    pub fn validate(self) -> Result<(), MtpError> {
        validate_chain_count(self.chains, 2)?;
        Ok(())
    }
}

/// Fitted MTP model metadata.
#[derive(Debug, Clone)]
pub struct MtpModel {
    /// Number of covariates in the binary design matrix.
    pub n_binary_covariates: usize,
    /// Number of covariates in the marginal-mean design matrix.
    pub n_mean_covariates: usize,
    /// Random-effects dimensionality per subject.
    pub random_effect_dimension: usize,
    /// Fit diagnostics.
    pub report: MtpReport,
}

/// MTP predictions.
///
/// NOTE: Full longitudinal prediction for MTP is complex and involves
/// integration over random effects. This is currently a placeholder.
#[derive(Debug, Clone)]
pub struct MtpPrediction {
    pub expected_outcome: Mat<f64>,
}

impl MtpPrediction {
    #[must_use]
    pub fn new(nrows: usize) -> Self {
        Self {
            expected_outcome: Mat::zeros(nrows, 1),
        }
    }
}

/// Component-wise acceptance rates from the MCMC sampler, over all iterations.
///
/// `alpha`, `beta` and `omega_sq` are proposed together, so the three rates are equal.
#[derive(Debug, Clone, Copy, Default)]
pub struct MtpAcceptanceRates {
    pub alpha: f64,
    pub beta: f64,
    pub random_effects: f64,
    pub family_effects: f64,
    pub kappa: f64,
    pub omega_sq: f64,
}

/// Sampler diagnostics summary.
#[derive(Debug, Clone, Default)]
pub struct MtpSamplerDiagnostics {
    pub iterations_completed: usize,
    pub retained_draws: usize,
    pub acceptance_rates: Option<MtpAcceptanceRates>,
}

/// Output report from MTP fitting.
#[derive(Debug, Clone, Default)]
pub struct MtpReport {
    /// Standardized fit metadata.
    pub meta: FitMetadata,
    pub diagnostics: MtpSamplerDiagnostics,
    pub posterior_summary: Option<MtpPosteriorSummary>,
}

impl Model for MtpModel {
    type Prediction = MtpPrediction;
    type Report = MtpReport;

    fn predict(&self, x: &Mat<f64>) -> Self::Prediction {
        let mut out = MtpPrediction::new(x.nrows());
        self.predict_into(x, &mut out);
        out
    }

    fn predict_into(&self, _x: &Mat<f64>, _out: &mut Self::Prediction) {
        // Implementation requires posterior draws and integration.
        unimplemented!("MtpModel prediction not yet implemented via Model trait")
    }

    fn report(&self) -> &Self::Report {
        &self.report
    }
}

/// Multi-chain split-R-hat diagnostics summary.
#[derive(Debug, Clone, Default)]
pub struct MtpConvergenceSummary {
    /// Number of chains included.
    pub chain_count: usize,
    /// Draws per chain used after truncation to equal even length.
    pub draws_per_chain_used: usize,
    /// Split-R-hat for each alpha coefficient, `None` where a half-chain is constant.
    pub alpha_split_rhat: Vec<Option<f64>>,
    /// Split-R-hat for each beta coefficient, `None` where a half-chain is constant.
    pub beta_split_rhat: Vec<Option<f64>>,
    /// Split-R-hat for kappa.
    pub kappa_split_rhat: Option<f64>,
    /// Split-R-hat for omega squared.
    pub omega_sq_split_rhat: Option<f64>,
    /// Maximum split-R-hat across all tracked scalar parameters, `None` when any of
    /// them is undefined.
    pub max_split_rhat: Option<f64>,
}

impl MtpConvergenceSummary {
    /// Whether every tracked parameter has a defined split-R-hat below `threshold`.
    /// A parameter whose chains are constant is not converged.
    #[must_use]
    pub fn is_converged(&self, threshold: f64) -> bool {
        self.max_split_rhat.is_some_and(|max| max < threshold)
    }
}

/// Output report for multi-chain MTP fitting.
#[derive(Debug, Clone, Default)]
pub struct MtpMultiChainReport {
    /// Chain-specific reports in execution order.
    pub chain_reports: Vec<MtpReport>,
    /// Posterior summary from pooled draws across all chains.
    pub pooled_posterior_summary: Option<MtpPosteriorSummary>,
    /// Convergence diagnostics from split-R-hat summaries.
    pub convergence: MtpConvergenceSummary,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_options_validate_and_retained_draws() {
        let options = MtpFitOptions::builder().iterations(100).burn_in(20).build();
        assert!(options.validate().is_ok());
        assert_eq!(options.retained_draws(), 20);
    }

    #[test]
    fn random_effect_dimension_matches_structure() {
        assert_eq!(
            RandomEffectsStructure::InterceptsOnly.random_effect_dimension(),
            2
        );
        assert_eq!(
            RandomEffectsStructure::InterceptsAndTimeSlopes.random_effect_dimension(),
            4
        );
    }

    #[test]
    fn multi_chain_options_validate() {
        let options = MtpMultiChainOptions { chains: 3 };
        assert!(options.validate().is_ok());
    }

    #[test]
    fn proposal_tuning_defaults_are_valid() {
        assert!(MtpProposalTuning::default().is_valid());
    }

    #[test]
    fn defaults_cover_extended_sampler_config() {
        let config = MtpSamplerConfig::default();
        assert_eq!(
            config.positive_part_distribution,
            PositivePartDistribution::LogSkewNormal
        );
        assert_eq!(config.family_random_effects, FamilyRandomEffects::Disabled);
    }

    #[test]
    fn sampler_config_validate_rejects_invalid_prior() {
        let config = MtpSamplerConfig::builder()
            .prior_config(MtpPriorConfig::builder().alpha_variance(0.0).build())
            .build();
        assert!(matches!(
            config.validate(),
            Err(MtpError::InvalidPriorConfig)
        ));
    }

    #[test]
    fn sampler_config_validate_rejects_invalid_proposal_tuning() {
        let config = MtpSamplerConfig::builder()
            .proposal_tuning(MtpProposalTuning::builder().adaptation_interval(0).build())
            .build();
        assert!(matches!(
            config.validate(),
            Err(MtpError::InvalidProposalTuning)
        ));
    }
}
