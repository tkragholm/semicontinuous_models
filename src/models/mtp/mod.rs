//! Correlated marginalized two-part (MTP) model scaffold.
//!
//! This module provides a structured, publication-ready API surface for
//! longitudinal MTP modeling.
//!
//! Current implementation supports correlated random intercepts and optional
//! time slopes, with posterior simulation workflows for downstream effects
//! and diagnostics.

pub mod diagnostics;
pub mod effects;
pub mod input;
pub mod likelihood;
pub mod posterior;
pub mod priors;
pub mod sampler;
pub mod types;

pub use diagnostics::{
    CalibrationBinSummary, PosteriorPredictiveSummary, autocorrelation, effective_sample_size,
    posterior_predictive_summary, summarize_multi_chain_convergence,
};
pub use effects::{
    CounterfactualEffects, CounterfactualEffectsSummary, CounterfactualScenario,
    EffectIntervalSummary, PeriodEffect, PeriodEffectSummary, compute_counterfactual_effects,
    compute_counterfactual_effects_summary,
};
pub use posterior::{
    MtpPosteriorDraw, MtpPosteriorSamples, MtpPosteriorSummary, ParameterSummary,
    summarize_posterior,
};
pub use priors::MtpPriorConfig;
pub use sampler::{
    fit_mtp_input, fit_mtp_input_multi_chain, fit_mtp_input_multi_chain_with_config,
    fit_mtp_input_multi_chain_with_posterior, fit_mtp_input_multi_chain_with_posterior_config,
    fit_mtp_input_with_config, fit_mtp_input_with_posterior, fit_mtp_input_with_posterior_config,
};
pub use types::{
    FamilyRandomEffects, MtpAcceptanceRates, MtpConvergenceSummary, MtpError, MtpFitOptions,
    MtpModel, MtpMultiChainOptions, MtpMultiChainReport, MtpProposalTuning, MtpReport,
    MtpSamplerConfig, MtpSamplerDiagnostics, PositivePartDistribution, RandomEffectsStructure,
};
