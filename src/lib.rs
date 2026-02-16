#![forbid(unsafe_code)]

//! # `semicontinuous_models`
//!
//! Reusable models and diagnostics for non-negative semi-continuous outcomes:
//! two-part, Tweedie GLM, and log-normal with smearing.
//!
//! The crate was initially developed for healthcare outcome analyses, but the API is
//! intentionally domain-agnostic and can be reused in other settings.

pub mod inference;
pub mod input;
pub mod models;
pub mod preprocess;
pub mod utils;

pub use inference::{InferenceError, McmcConfig, ProposalStats};
pub use input::{InputError, LongitudinalInputError, LongitudinalModelInput, ModelInput};
pub use preprocess::{
    OutcomeDiagnostics, column_has_variation, drop_constant_columns, outcome_diagnostics,
};
pub mod comparison {
    pub use crate::models::comparison::*;
}
pub mod matrix_ops {
    pub use crate::models::matrix_ops::*;
}

pub use models::two_part::{
    BootstrapOptions, BootstrapResult, BootstrapSummary, ConfidenceInterval, FitOptions,
    Regularization, TwoPartError, TwoPartModel, TwoPartPrediction, TwoPartReport,
    bootstrap as bootstrap_two_part, bootstrap_percentile_ci, bootstrap_summary,
    coefficient_confidence_intervals, fit_two_part_clustered_input, fit_two_part_input,
    fit_two_part_weighted_input, log_likelihood as two_part_log_likelihood,
};

pub use models::tweedie::{
    TweedieError, TweedieModel, TweedieOptions, TweediePrediction, TweedieReport,
    deviance as tweedie_deviance, fit_tweedie_input,
    quasi_log_likelihood as tweedie_quasi_log_likelihood,
};

pub use models::lognormal::{
    LogNormalError, LogNormalModel, LogNormalOptions, LogNormalPrediction, LogNormalReport,
    fit_lognormal_smearing_input, log_likelihood as lognormal_log_likelihood,
};

pub use models::selection::{
    CrossValidationError, CrossValidationOptions, CrossValidationResult, InformationCriteria,
    LogNormalCandidate, ModelFitMetrics, ParkTestResult, SelectionResult, TweedieCandidate,
    TweedieCvCandidate, compute_information_criteria, compute_model_fit_metrics,
    cross_validate_models_input, park_test, recommend_from_cv, select_models_input,
};

pub use models::comparison::{
    ComparisonTables, ModelComparison, ModelComparisonError, ModelComparisonOptions,
    ModelInformationCriteria, ModelScore, TweedieRankingRow, compare_models_input,
    render_comparison_tables,
};

pub use models::mtp::{
    CalibrationBinSummary, CounterfactualEffects, CounterfactualEffectsSummary,
    CounterfactualScenario, EffectIntervalSummary, FamilyRandomEffects, MtpAcceptanceRates,
    MtpConvergenceSummary, MtpError, MtpFitOptions, MtpModel, MtpMultiChainOptions,
    MtpMultiChainReport, MtpPosteriorDraw, MtpPosteriorSamples, MtpPosteriorSummary,
    MtpPriorConfig, MtpProposalTuning, MtpReport, MtpSamplerConfig, MtpSamplerDiagnostics,
    ParameterSummary, PeriodEffect, PeriodEffectSummary, PositivePartDistribution,
    PosteriorPredictiveSummary, RandomEffectsStructure, autocorrelation,
    compute_counterfactual_effects, compute_counterfactual_effects_summary, effective_sample_size,
    fit_mtp_input, fit_mtp_input_multi_chain, fit_mtp_input_multi_chain_with_config,
    fit_mtp_input_multi_chain_with_posterior, fit_mtp_input_multi_chain_with_posterior_config,
    fit_mtp_input_with_config, fit_mtp_input_with_posterior, fit_mtp_input_with_posterior_config,
    posterior_predictive_summary, summarize_multi_chain_convergence, summarize_posterior,
};
