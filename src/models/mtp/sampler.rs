//! Sampler entrypoints for correlated longitudinal MTP.

use crate::input::LongitudinalModelInput;
use crate::utils::{acceptance_rate, weighted_xtx, weighted_xtz};
use faer::Mat;
use std::time::Instant;

use super::diagnostics::summarize_multi_chain_convergence;
use super::input::{SubjectRows, prepare_input};
use super::posterior::{MtpPosteriorSamples, summarize_posterior};
use super::types::{
    FamilyRandomEffects, MtpAcceptanceRates, MtpError, MtpFitOptions, MtpModel,
    MtpMultiChainOptions, MtpMultiChainReport, MtpReport, MtpSamplerConfig, MtpSamplerDiagnostics,
    PositivePartDistribution, RandomEffectsStructure,
};
use super::{likelihood, priors};

mod chain;
mod context;
mod math;
mod updates;

#[allow(clippy::wildcard_imports)]
use self::chain::*;
#[allow(clippy::wildcard_imports)]
use self::math::*;

const MIN_WEIGHT: f64 = 1.0e-8;
const RIDGE_L2: f64 = 1.0e-8;
const RE_UPDATE_TOLERANCE: f64 = 1.0e-6;
const OUTER_TOLERANCE: f64 = 1.0e-5;
const SUBJECT_NEWTON_MAX_ITERS: usize = 8;

#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_weighted_xtx(x: &Mat<f64>, weights: &Mat<f64>) -> Mat<f64> {
    weighted_xtx(x, weights)
}

#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_weighted_xtz(x: &Mat<f64>, weights: &Mat<f64>, response: &Mat<f64>) -> Mat<f64> {
    weighted_xtz(x, weights, response)
}

#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_sample_standard_normals(count: usize, seed: u64) -> f64 {
    use rand::{SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sum = 0.0;
    for _ in 0..count {
        sum += sample_standard_normal(&mut rng);
    }
    sum
}

#[derive(Debug, Clone)]
struct BaselineEstimate {
    alpha: Vec<f64>,
    beta: Vec<f64>,
    subject_effects: Vec<Vec<f64>>,
    random_effects_cov: Mat<f64>,
    kappa: f64,
    omega_sq: f64,
    structure: RandomEffectsStructure,
}

#[derive(Debug, Clone, Copy)]
enum RandomBlock {
    Binary,
    Mean,
}

#[derive(Debug, Clone)]
struct InitialEstimationState {
    binary_outcome: Mat<f64>,
    positive_indices: Vec<usize>,
    x_mean_positive: Mat<f64>,
    log_outcome_positive: Mat<f64>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
}

struct SubjectUpdateContext<'a> {
    input: &'a LongitudinalModelInput,
    alpha: &'a [f64],
    beta: &'a [f64],
    omega_sq: f64,
    prior_precision: &'a Mat<f64>,
    structure: RandomEffectsStructure,
}

#[derive(Debug, Clone)]
struct ChainState {
    alpha: Vec<f64>,
    beta: Vec<f64>,
    subject_effects: Vec<Vec<f64>>,
    random_effects_cov: Mat<f64>,
    random_effects_precision: Mat<f64>,
    family_effects: Vec<[f64; 2]>,
    kappa: f64,
    omega_sq: f64,
}

#[derive(Debug, Clone)]
struct ProposalScales {
    alpha: Vec<f64>,
    beta: Vec<f64>,
    random_effects: Vec<f64>,
    family_effects: [f64; 2],
    kappa: f64,
    log_omega_sq: f64,
}

struct SamplerContext<'a> {
    input: &'a LongitudinalModelInput,
    subjects: &'a [SubjectRows],
    row_to_subject: &'a [usize],
    row_to_family: Option<&'a [usize]>,
    structure: RandomEffectsStructure,
    prior_config: priors::MtpPriorConfig,
    positive_part_distribution: PositivePartDistribution,
    family_random_effects: FamilyRandomEffects,
}

struct SamplingResult {
    samples: MtpPosteriorSamples,
    acceptance_rates: MtpAcceptanceRates,
}

#[derive(Debug, Clone)]
struct RowLikelihoodCache {
    row_log_likelihood: Vec<f64>,
    binary_fixed: Vec<f64>,
    mean_fixed: Vec<f64>,
    binary_offset: Vec<f64>,
    mean_offset: Vec<f64>,
    total: f64,
}

#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, Copy)]
struct PosteriorCache {
    log_likelihood: f64,
    log_prior_alpha: f64,
    log_prior_beta: f64,
    log_prior_random_effects: f64,
    log_prior_family: f64,
    log_prior_kappa: f64,
    log_prior_omega: f64,
}

#[derive(Debug, Clone)]
struct SubjectProposalEvaluation {
    candidate_row_values: Vec<f64>,
    current_sum: f64,
    candidate_sum: f64,
    prior_delta: f64,
}

#[derive(Debug, Clone)]
struct SamplerBuffers {
    alpha_proposal: Vec<f64>,
    beta_proposal: Vec<f64>,
    subject_proposals: Vec<Vec<f64>>,
}

impl PosteriorCache {
    fn initialize_with_likelihood(
        context: &SamplerContext<'_>,
        state: &ChainState,
        log_likelihood: f64,
    ) -> Result<Self, MtpError> {
        if !context.validate_state(state) {
            return Err(MtpError::NonConvergence);
        }
        if !log_likelihood.is_finite() {
            return Err(MtpError::NonConvergence);
        }

        let cache = Self {
            log_likelihood,
            log_prior_alpha: context.log_alpha_prior(&state.alpha),
            log_prior_beta: context.log_beta_prior(&state.beta),
            log_prior_random_effects: SamplerContext::log_subject_random_effect_prior(state),
            log_prior_family: context.log_family_prior(&state.family_effects),
            log_prior_kappa: context.log_kappa_prior(state.kappa),
            log_prior_omega: context.log_omega_prior(state.omega_sq),
        };

        if !cache.total().is_finite() {
            return Err(MtpError::NonConvergence);
        }

        Ok(cache)
    }

    const fn total(self) -> f64 {
        self.log_likelihood
            + self.log_prior_alpha
            + self.log_prior_beta
            + self.log_prior_random_effects
            + self.log_prior_family
            + self.log_prior_kappa
            + self.log_prior_omega
    }
}

impl RowLikelihoodCache {
    fn initialize(context: &SamplerContext<'_>, state: &ChainState) -> Result<Self, MtpError> {
        let binary_fixed = linear_predictor(&context.input.x_binary, &state.alpha);
        let mean_fixed = linear_predictor(&context.input.x_mean, &state.beta);
        let (binary_offset, mean_offset) = context.row_offsets(state);
        let row_log_likelihood = context
            .recompute_all_rows_from_offsets(
                state,
                &binary_fixed,
                &mean_fixed,
                &binary_offset,
                &mean_offset,
            )
            .ok_or(MtpError::NonConvergence)?;
        let total = row_log_likelihood.iter().sum::<f64>();

        if !total.is_finite() {
            return Err(MtpError::NonConvergence);
        }

        Ok(Self {
            row_log_likelihood,
            binary_fixed,
            mean_fixed,
            binary_offset,
            mean_offset,
            total,
        })
    }

    fn sum_rows(&self, rows: &[usize]) -> f64 {
        rows.iter().map(|row| self.row_log_likelihood[*row]).sum()
    }

    fn replace_rows(&mut self, rows: &[usize], replacement: &[f64]) -> f64 {
        debug_assert_eq!(rows.len(), replacement.len());
        let mut delta = 0.0;
        for (idx, row) in rows.iter().copied().enumerate() {
            let previous = self.row_log_likelihood[row];
            let next = replacement[idx];
            self.row_log_likelihood[row] = next;
            delta += next - previous;
        }
        self.total += delta;
        delta
    }

    fn replace_all_rows_with_binary(
        &mut self,
        row_log_likelihood: Vec<f64>,
        binary_fixed: Vec<f64>,
    ) {
        self.total = row_log_likelihood.iter().sum();
        self.row_log_likelihood = row_log_likelihood;
        self.binary_fixed = binary_fixed;
    }

    fn replace_all_rows_with_mean(&mut self, row_log_likelihood: Vec<f64>, mean_fixed: Vec<f64>) {
        self.total = row_log_likelihood.iter().sum();
        self.row_log_likelihood = row_log_likelihood;
        self.mean_fixed = mean_fixed;
    }

    fn update_subject_offsets(
        &mut self,
        time: &[f64],
        rows: &[usize],
        structure: RandomEffectsStructure,
        current_effect: &[f64],
        next_effect: &[f64],
    ) {
        for &row in rows {
            let row_time = time[row];
            self.binary_offset[row] += random_binary_component(next_effect, row_time, structure)
                - random_binary_component(current_effect, row_time, structure);
            self.mean_offset[row] += random_mean_component(next_effect, row_time, structure)
                - random_mean_component(current_effect, row_time, structure);
        }
    }

    fn update_family_offsets(&mut self, rows: &[usize], delta_binary: f64, delta_mean: f64) {
        for &row in rows {
            self.binary_offset[row] += delta_binary;
            self.mean_offset[row] += delta_mean;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PosteriorSimulationRequest<'a> {
    input: &'a LongitudinalModelInput,
    subjects: &'a [SubjectRows],
    row_to_subject: &'a [usize],
    row_to_family: Option<&'a [usize]>,
    family_count: usize,
    baseline: &'a BaselineEstimate,
    config: MtpSamplerConfig,
    retained_draws: usize,
}

#[derive(Default)]
struct AcceptanceCounts {
    accepted_alpha: usize,
    proposed_alpha: usize,
    accepted_beta: usize,
    proposed_beta: usize,
    accepted_random_effects: usize,
    proposed_random_effects: usize,
    accepted_family_effects: usize,
    proposed_family_effects: usize,
    accepted_kappa: usize,
    proposed_kappa: usize,
    accepted_omega: usize,
    proposed_omega: usize,
}

impl AcceptanceCounts {
    fn record_alpha(&mut self, accepted: bool) {
        self.proposed_alpha += 1;
        self.accepted_alpha += usize::from(accepted);
    }

    fn record_beta(&mut self, accepted: bool) {
        self.proposed_beta += 1;
        self.accepted_beta += usize::from(accepted);
    }

    const fn record_random_effects(&mut self, accepted: usize, proposed: usize) {
        self.proposed_random_effects += proposed;
        self.accepted_random_effects += accepted;
    }

    const fn record_family_effects(&mut self, accepted: usize, proposed: usize) {
        self.proposed_family_effects += proposed;
        self.accepted_family_effects += accepted;
    }

    fn record_kappa(&mut self, accepted: bool) {
        self.proposed_kappa += 1;
        self.accepted_kappa += usize::from(accepted);
    }

    fn record_omega(&mut self, accepted: bool) {
        self.proposed_omega += 1;
        self.accepted_omega += usize::from(accepted);
    }

    fn rates(&self) -> MtpAcceptanceRates {
        MtpAcceptanceRates {
            alpha: acceptance_rate(self.accepted_alpha, self.proposed_alpha),
            beta: acceptance_rate(self.accepted_beta, self.proposed_beta),
            random_effects: acceptance_rate(
                self.accepted_random_effects,
                self.proposed_random_effects,
            ),
            family_effects: acceptance_rate(
                self.accepted_family_effects,
                self.proposed_family_effects,
            ),
            kappa: acceptance_rate(self.accepted_kappa, self.proposed_kappa),
            omega_sq: acceptance_rate(self.accepted_omega, self.proposed_omega),
        }
    }
}

/// Fit the correlated MTP model from longitudinal input.
///
/// Current behavior:
/// - supports random intercepts and random time slopes,
/// - estimates correlated subject random effects,
/// - returns model metadata and posterior summary.
///
/// # Errors
///
/// Returns `MtpError` if inputs/options are invalid.
pub fn fit_mtp_input(
    input: &LongitudinalModelInput,
    options: MtpFitOptions,
) -> Result<(MtpModel, MtpReport), MtpError> {
    let config = MtpSamplerConfig {
        fit_options: options,
        ..MtpSamplerConfig::default()
    };
    let (model, report, _) = fit_mtp_input_with_posterior_config(input, config)?;
    Ok((model, report))
}

/// Fit the correlated MTP model with explicit prior and proposal configuration.
///
/// # Errors
///
/// Returns `MtpError` if inputs/options are invalid.
pub fn fit_mtp_input_with_config(
    input: &LongitudinalModelInput,
    config: MtpSamplerConfig,
) -> Result<(MtpModel, MtpReport), MtpError> {
    let (model, report, _) = fit_mtp_input_with_posterior_config(input, config)?;
    Ok((model, report))
}

/// Fit MTP and return posterior draws.
///
/// This is useful for downstream effect calculations using
/// `compute_counterfactual_effects`.
///
/// # Errors
///
/// Returns `MtpError` if input/options are invalid.
pub fn fit_mtp_input_with_posterior(
    input: &LongitudinalModelInput,
    options: MtpFitOptions,
) -> Result<(MtpModel, MtpReport, MtpPosteriorSamples), MtpError> {
    let config = MtpSamplerConfig {
        fit_options: options,
        ..MtpSamplerConfig::default()
    };
    fit_mtp_input_with_posterior_config(input, config)
}

/// Fit MTP and return posterior draws with explicit prior/proposal settings.
///
/// # Errors
///
/// Returns `MtpError` if input/options are invalid.
pub fn fit_mtp_input_with_posterior_config(
    input: &LongitudinalModelInput,
    config: MtpSamplerConfig,
) -> Result<(MtpModel, MtpReport, MtpPosteriorSamples), MtpError> {
    let start_time = Instant::now();
    config.validate()?;
    if config.family_random_effects != FamilyRandomEffects::Disabled && input.family_ids.is_none() {
        return Err(MtpError::MissingFamilyIds);
    }
    let options = config.fit_options;

    let prepared = prepare_input(input)?;
    let subject_count = prepared.n_subjects();
    let subject_checksum = prepared
        .subjects
        .iter()
        .map(|subject| subject.subject_id)
        .sum::<u64>();
    debug_assert_eq!(subject_count, prepared.subjects.len());
    let _ = subject_checksum;

    let row_to_subject =
        build_row_to_subject_map(prepared.input.outcome.nrows(), &prepared.subjects);
    let family_mapping = if config.family_random_effects == FamilyRandomEffects::Disabled {
        None
    } else {
        input
            .family_ids
            .as_ref()
            .map(|family_ids| build_row_to_group_map(family_ids))
    };
    let (row_to_family, family_count) = family_mapping
        .as_ref()
        .map_or((None, 0), |(mapping, count)| {
            (Some(mapping.as_slice()), *count)
        });
    let baseline = estimate_correlated_random_effects(&prepared, &row_to_subject, options)?;

    let retained_draws = options.retained_draws();
    let sampling = simulate_posterior_draws(&PosteriorSimulationRequest {
        input: prepared.input,
        subjects: &prepared.subjects,
        row_to_subject: &row_to_subject,
        row_to_family,
        family_count,
        baseline: &baseline,
        config,
        retained_draws,
    })?;
    let posterior = sampling.samples;

    let posterior_summary = if posterior.is_empty() {
        None
    } else {
        Some(summarize_posterior(&posterior))
    };

    let execution_time = start_time.elapsed();
    let meta = crate::models::FitMetadata {
        iterations: options.iterations,
        converged: true,
        execution_time,
        solver: crate::models::SolverKind::Mcmc,
        ..crate::models::FitMetadata::default()
    };

    let report = MtpReport {
        meta,
        diagnostics: MtpSamplerDiagnostics {
            iterations_completed: options.iterations,
            retained_draws,
            acceptance_rates: Some(sampling.acceptance_rates),
        },
        posterior_summary,
    };

    let model = MtpModel {
        n_binary_covariates: prepared.input.x_binary.ncols(),
        n_mean_covariates: prepared.input.x_mean.ncols(),
        random_effect_dimension: options.random_effects.random_effect_dimension(),
        report: report.clone(),
    };

    Ok((model, report, posterior))
}

/// Fit MTP using multiple independent chains and return pooled summaries.
///
/// # Errors
///
/// Returns `MtpError` if input/options are invalid or chain fitting fails.
pub fn fit_mtp_input_multi_chain(
    input: &LongitudinalModelInput,
    options: MtpFitOptions,
    multi_chain: MtpMultiChainOptions,
) -> Result<(MtpModel, MtpMultiChainReport), MtpError> {
    let config = MtpSamplerConfig {
        fit_options: options,
        ..MtpSamplerConfig::default()
    };
    let (model, report, _) =
        fit_mtp_input_multi_chain_with_posterior_config(input, config, multi_chain)?;
    Ok((model, report))
}

/// Fit MTP with explicit prior/proposal settings using multiple independent chains.
///
/// # Errors
///
/// Returns `MtpError` if input/options are invalid or chain fitting fails.
pub fn fit_mtp_input_multi_chain_with_config(
    input: &LongitudinalModelInput,
    config: MtpSamplerConfig,
    multi_chain: MtpMultiChainOptions,
) -> Result<(MtpModel, MtpMultiChainReport), MtpError> {
    let (model, report, _) =
        fit_mtp_input_multi_chain_with_posterior_config(input, config, multi_chain)?;
    Ok((model, report))
}

/// Fit MTP using multiple independent chains and return chain-wise posterior draws.
///
/// # Errors
///
/// Returns `MtpError` if input/options are invalid or chain fitting fails.
pub fn fit_mtp_input_multi_chain_with_posterior(
    input: &LongitudinalModelInput,
    options: MtpFitOptions,
    multi_chain: MtpMultiChainOptions,
) -> Result<(MtpModel, MtpMultiChainReport, Vec<MtpPosteriorSamples>), MtpError> {
    let config = MtpSamplerConfig {
        fit_options: options,
        ..MtpSamplerConfig::default()
    };
    fit_mtp_input_multi_chain_with_posterior_config(input, config, multi_chain)
}

/// Fit MTP with explicit prior/proposal settings using multiple independent chains.
///
/// # Errors
///
/// Returns `MtpError` if input/options are invalid or chain fitting fails.
#[allow(clippy::too_many_lines)]
pub fn fit_mtp_input_multi_chain_with_posterior_config(
    input: &LongitudinalModelInput,
    config: MtpSamplerConfig,
    multi_chain: MtpMultiChainOptions,
) -> Result<(MtpModel, MtpMultiChainReport, Vec<MtpPosteriorSamples>), MtpError> {
    let start_time = Instant::now();
    config.validate()?;
    multi_chain.validate()?;
    if config.family_random_effects != FamilyRandomEffects::Disabled && input.family_ids.is_none() {
        return Err(MtpError::MissingFamilyIds);
    }
    let options = config.fit_options;

    let prepared = prepare_input(input)?;
    let row_to_subject =
        build_row_to_subject_map(prepared.input.outcome.nrows(), &prepared.subjects);
    let family_mapping = if config.family_random_effects == FamilyRandomEffects::Disabled {
        None
    } else {
        input
            .family_ids
            .as_ref()
            .map(|family_ids| build_row_to_group_map(family_ids))
    };
    let (row_to_family, family_count) = family_mapping
        .as_ref()
        .map_or((None, 0), |(mapping, count)| {
            (Some(mapping.as_slice()), *count)
        });
    let baseline = estimate_correlated_random_effects(&prepared, &row_to_subject, options)?;

    let retained_draws = options.retained_draws();
    let mut chain_reports = Vec::with_capacity(multi_chain.chains);
    let mut chain_posteriors = Vec::with_capacity(multi_chain.chains);
    let prepared_input = prepared.input;
    let prepared_subjects = prepared.subjects.as_slice();
    let row_to_subject_ref = row_to_subject.as_slice();
    let baseline_ref = &baseline;
    let mut chain_results = (0..multi_chain.chains)
        .map(|_| None)
        .collect::<Vec<Option<Result<SamplingResult, MtpError>>>>();

    std::thread::scope(|scope| -> Result<(), MtpError> {
        let mut handles = Vec::with_capacity(multi_chain.chains);
        for chain_index in 0..multi_chain.chains {
            let mut chain_options = options;
            let index_u64 = u64::try_from(chain_index).unwrap_or(u64::MAX);
            chain_options.seed = options
                .seed
                .wrapping_add(index_u64.saturating_mul(multi_chain.seed_stride));
            let chain_config = MtpSamplerConfig {
                fit_options: chain_options,
                ..config
            };

            handles.push((
                chain_index,
                chain_options,
                scope.spawn(move || {
                    simulate_posterior_draws(&PosteriorSimulationRequest {
                        input: prepared_input,
                        subjects: prepared_subjects,
                        row_to_subject: row_to_subject_ref,
                        row_to_family,
                        family_count,
                        baseline: baseline_ref,
                        config: chain_config,
                        retained_draws,
                    })
                }),
            ));
        }

        for (chain_index, _chain_options, handle) in handles {
            let result = handle.join().map_err(|_| MtpError::NonConvergence)?;
            chain_results[chain_index] = Some(result);
        }

        Ok(())
    })?;

    for (chain_index, chain_result) in chain_results
        .iter_mut()
        .enumerate()
        .take(multi_chain.chains)
    {
        let mut chain_options = options;
        let index_u64 = u64::try_from(chain_index).unwrap_or(u64::MAX);
        chain_options.seed = options
            .seed
            .wrapping_add(index_u64.saturating_mul(multi_chain.seed_stride));

        let sampling = chain_result.take().ok_or(MtpError::NonConvergence)??;

        let posterior_summary = if sampling.samples.is_empty() {
            None
        } else {
            Some(summarize_posterior(&sampling.samples))
        };
        let execution_time_approx = start_time.elapsed();
        let meta = crate::models::FitMetadata {
            iterations: chain_options.iterations,
            converged: true,
            execution_time: execution_time_approx,
            solver: crate::models::SolverKind::Mcmc,
            ..crate::models::FitMetadata::default()
        };
        chain_reports.push(MtpReport {
            meta,
            diagnostics: MtpSamplerDiagnostics {
                iterations_completed: chain_options.iterations,
                retained_draws,
                acceptance_rates: Some(sampling.acceptance_rates),
            },
            posterior_summary,
        });
        chain_posteriors.push(sampling.samples);
    }

    let pooled = combine_posteriors(&chain_posteriors);
    let pooled_posterior_summary = if pooled.is_empty() {
        None
    } else {
        Some(summarize_posterior(&pooled))
    };
    let convergence = summarize_multi_chain_convergence(&chain_posteriors)?;

    let model = MtpModel {
        n_binary_covariates: prepared.input.x_binary.ncols(),
        n_mean_covariates: prepared.input.x_mean.ncols(),
        random_effect_dimension: options.random_effects.random_effect_dimension(),
        report: chain_reports[0].clone(), // Use first chain for base report
    };

    Ok((
        model,
        MtpMultiChainReport {
            chain_reports,
            pooled_posterior_summary,
            convergence,
        },
        chain_posteriors,
    ))
}

#[cfg(test)]
mod tests {
    use faer::Mat;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::updates::update_alpha_block;
    use super::*;
    use crate::utils::{matrix_is_finite, usize_to_f64, weighted_xtx, weighted_xtz};

    fn basic_input() -> LongitudinalModelInput {
        LongitudinalModelInput::new(
            Mat::from_fn(12, 1, |row, _| {
                let subject = row / 3;
                let subj_intercept = if subject < 2 { -0.5 } else { 0.8 };
                let subj_slope = if subject % 2 == 0 { -0.2 } else { 0.2 };
                let time = usize_to_f64(row % 3);
                if row % 3 == 0 {
                    0.0
                } else {
                    (2.0f64 + subj_intercept + subj_slope * time).exp()
                }
            }),
            Mat::from_fn(
                12,
                2,
                |row, col| {
                    if col == 0 { 1.0 } else { usize_to_f64(row % 3) }
                },
            ),
            Mat::from_fn(
                12,
                2,
                |row, col| {
                    if col == 0 { 1.0 } else { usize_to_f64(row % 3) }
                },
            ),
            vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        )
    }

    #[test]
    fn fit_returns_model_and_report_for_intercepts() {
        let input = basic_input();
        let options = MtpFitOptions::builder().iterations(80).burn_in(20).build();

        let (model, report) = fit_mtp_input(&input, options).expect("fit should run");
        assert_eq!(model.random_effect_dimension, 2);
        assert_eq!(report.diagnostics.retained_draws, 15);
        assert!(report.posterior_summary.is_some());
        assert!(report.diagnostics.acceptance_rates.is_some());
    }

    #[test]
    fn fit_returns_model_and_report_for_slopes() {
        let input = basic_input();
        let options = MtpFitOptions::builder()
            .iterations(80)
            .burn_in(20)
            .random_effects(RandomEffectsStructure::InterceptsAndTimeSlopes)
            .build();

        let (model, report) = fit_mtp_input(&input, options).expect("fit should run");
        assert_eq!(model.random_effect_dimension, 4);
        assert_eq!(report.diagnostics.retained_draws, 15);
        assert!(report.posterior_summary.is_some());
        assert!(report.diagnostics.acceptance_rates.is_some());
    }

    #[test]
    fn fit_with_posterior_returns_draws() {
        let input = basic_input();
        let options = MtpFitOptions::builder()
            .iterations(60)
            .burn_in(20)
            .thin(2)
            .build();

        let (_, report, posterior) =
            fit_mtp_input_with_posterior(&input, options).expect("posterior should run");
        assert_eq!(posterior.len(), report.diagnostics.retained_draws);
    }

    #[test]
    fn multi_chain_fit_returns_convergence_summary() {
        let input = basic_input();
        let options = MtpFitOptions::builder()
            .iterations(60)
            .burn_in(20)
            .thin(2)
            .build();
        let multi_chain = MtpMultiChainOptions {
            chains: 2,
            seed_stride: 7,
        };

        let (model, report, chains) =
            fit_mtp_input_multi_chain_with_posterior(&input, options, multi_chain)
                .expect("multi-chain fit should run");

        assert_eq!(model.random_effect_dimension, 2);
        assert_eq!(report.chain_reports.len(), 2);
        assert_eq!(chains.len(), 2);
        assert_eq!(report.convergence.chain_count, 2);
        assert!(report.convergence.max_split_rhat.is_some());
    }

    #[test]
    fn first_multi_chain_matches_single_chain_seed_path() {
        let input = basic_input();
        let config = MtpSamplerConfig {
            fit_options: MtpFitOptions::builder()
                .iterations(70)
                .burn_in(20)
                .thin(2)
                .seed(1_337)
                .build(),
            ..MtpSamplerConfig::default()
        };

        let (_, _, single_posterior) = fit_mtp_input_with_posterior_config(&input, config)
            .expect("single-chain fit should run");
        let (_, _, chains) = fit_mtp_input_multi_chain_with_posterior_config(
            &input,
            config,
            MtpMultiChainOptions {
                chains: 2,
                seed_stride: 7,
            },
        )
        .expect("multi-chain fit should run");

        assert_eq!(chains.len(), 2);
        assert_eq!(chains[0].len(), single_posterior.len());
        for (single_draw, chain_draw) in single_posterior.draws.iter().zip(&chains[0].draws) {
            for (single_alpha, chain_alpha) in single_draw.alpha.iter().zip(&chain_draw.alpha) {
                assert!((single_alpha - chain_alpha).abs() < 1.0e-12);
            }
            for (single_beta, chain_beta) in single_draw.beta.iter().zip(&chain_draw.beta) {
                assert!((single_beta - chain_beta).abs() < 1.0e-12);
            }
            assert!((single_draw.kappa - chain_draw.kappa).abs() < 1.0e-12);
            assert!((single_draw.omega_sq - chain_draw.omega_sq).abs() < 1.0e-12);
        }
    }

    #[test]
    fn fit_with_config_applies_custom_kappa_bounds() {
        let input = basic_input();
        let config = MtpSamplerConfig {
            fit_options: MtpFitOptions::builder()
                .iterations(60)
                .burn_in(20)
                .thin(2)
                .build(),
            prior_config: priors::MtpPriorConfig {
                kappa_lower: -0.2,
                kappa_upper: 0.2,
                ..priors::MtpPriorConfig::default()
            },
            ..MtpSamplerConfig::default()
        };

        let (_, _, posterior) =
            fit_mtp_input_with_posterior_config(&input, config).expect("fit should run");
        assert!(!posterior.draws.is_empty());
        assert!(
            posterior
                .draws
                .iter()
                .all(|draw| draw.kappa >= -0.2 && draw.kappa <= 0.2)
        );
    }

    #[test]
    fn lognormal_positive_part_keeps_kappa_fixed_at_zero() {
        let input = basic_input();
        let config = MtpSamplerConfig {
            fit_options: MtpFitOptions::builder()
                .iterations(60)
                .burn_in(20)
                .thin(2)
                .build(),
            positive_part_distribution: PositivePartDistribution::LogNormal,
            ..MtpSamplerConfig::default()
        };

        let (_, _, posterior) =
            fit_mtp_input_with_posterior_config(&input, config).expect("fit should run");
        assert!(posterior.draws.iter().all(|draw| draw.kappa == 0.0));
    }

    #[test]
    fn family_layer_requires_family_ids() {
        let input = basic_input();
        let config = MtpSamplerConfig {
            family_random_effects: FamilyRandomEffects::InterceptsOnly,
            ..MtpSamplerConfig::default()
        };
        let result = fit_mtp_input_with_posterior_config(&input, config);
        assert!(matches!(result, Err(MtpError::MissingFamilyIds)));
    }

    #[test]
    fn family_layer_runs_when_family_ids_are_present() {
        let family_ids = vec![10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20];
        let input = basic_input().with_family_ids(family_ids);
        let config = MtpSamplerConfig {
            fit_options: MtpFitOptions::builder()
                .iterations(60)
                .burn_in(20)
                .thin(2)
                .build(),
            family_random_effects: FamilyRandomEffects::InterceptsOnly,
            ..MtpSamplerConfig::default()
        };

        let (_, report, posterior) =
            fit_mtp_input_with_posterior_config(&input, config).expect("fit should run");
        assert!(!posterior.draws.is_empty());
        let rates = report
            .diagnostics
            .acceptance_rates
            .expect("acceptance rates should be present");
        assert!(rates.family_effects > 0.0);
    }

    #[test]
    fn random_effect_covariance_dimension_tracks_structure() {
        let input = basic_input();
        let prepared = prepare_input(&input).expect("input should be valid");
        let row_to_subject =
            build_row_to_subject_map(prepared.input.outcome.nrows(), &prepared.subjects);

        let estimate = estimate_correlated_random_effects(
            &prepared,
            &row_to_subject,
            MtpFitOptions::builder()
                .iterations(40)
                .burn_in(10)
                .thin(2)
                .random_effects(RandomEffectsStructure::InterceptsAndTimeSlopes)
                .build(),
        )
        .expect("estimation should run");

        assert_eq!(estimate.random_effects_cov.ncols(), 4);
        assert!(matrix_is_finite(&estimate.random_effects_cov));
    }

    #[test]
    fn row_likelihood_cache_matches_full_likelihood() {
        let input = basic_input();
        let prepared = prepare_input(&input).expect("input should be valid");
        let row_to_subject =
            build_row_to_subject_map(prepared.input.outcome.nrows(), &prepared.subjects);
        let baseline = estimate_correlated_random_effects(
            &prepared,
            &row_to_subject,
            MtpFitOptions::builder()
                .iterations(40)
                .burn_in(10)
                .thin(2)
                .build(),
        )
        .expect("baseline should run");

        let context = SamplerContext {
            input: prepared.input,
            subjects: &prepared.subjects,
            row_to_subject: &row_to_subject,
            row_to_family: None,
            structure: baseline.structure,
            prior_config: priors::MtpPriorConfig::default(),
            positive_part_distribution: PositivePartDistribution::LogSkewNormal,
            family_random_effects: FamilyRandomEffects::Disabled,
        };
        let mut state = ChainState {
            alpha: baseline.alpha,
            beta: baseline.beta,
            subject_effects: baseline.subject_effects,
            random_effects_cov: baseline.random_effects_cov.clone(),
            random_effects_precision: invert_matrix_with_jitter(&baseline.random_effects_cov),
            family_effects: Vec::new(),
            kappa: baseline.kappa,
            omega_sq: baseline.omega_sq,
        };

        let mut row_cache =
            RowLikelihoodCache::initialize(&context, &state).expect("cache should build");
        let full = context.log_likelihood(&state);
        assert!((row_cache.total - full).abs() < 1.0e-8);

        let mut posterior =
            PosteriorCache::initialize_with_likelihood(&context, &state, row_cache.total)
                .expect("posterior cache should build");
        let mut rng = StdRng::seed_from_u64(42);
        let alpha_scales = vec![0.05; state.alpha.len()];
        let mut alpha_buffer = vec![0.0; state.alpha.len()];
        let _ = update_alpha_block(
            &context,
            &mut rng,
            &mut state,
            &mut posterior,
            &mut row_cache,
            &mut alpha_buffer,
            &alpha_scales,
            1.0e-3,
        );
        let recomputed = context.log_likelihood(&state);
        assert!((row_cache.total - recomputed).abs() < 1.0e-8);
        assert!((posterior.log_likelihood - recomputed).abs() < 1.0e-8);
    }

    #[test]
    fn weighted_xtx_matches_diagonal_reference() {
        let n = 9usize;
        let p = 4usize;
        let x = Mat::from_fn(n, p, |row, col| {
            let row_f = usize_to_f64(row);
            let col_f = usize_to_f64(col);
            (0.3f64.mul_add(row_f, 0.7 * col_f)).sin()
        });
        let weights = Mat::from_fn(n, 1, |row, _| {
            0.9f64.mul_add((0.2f64 * usize_to_f64(row)).cos().abs(), 0.1)
        });

        let optimized = weighted_xtx(&x, &weights);
        let diagonal = Mat::from_fn(
            n,
            n,
            |row, col| if row == col { weights[(row, 0)] } else { 0.0 },
        );
        let reference = x.transpose() * diagonal * x;

        for row in 0..p {
            for col in 0..p {
                assert!((optimized[(row, col)] - reference[(row, col)]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn weighted_xtz_matches_diagonal_reference() {
        let n = 11usize;
        let p = 3usize;
        let x = Mat::from_fn(n, p, |row, col| {
            let row_f = usize_to_f64(row);
            let col_f = usize_to_f64(col);
            (0.1f64.mul_add(row_f, -(0.4 * col_f))).cos()
        });
        let weights = Mat::from_fn(n, 1, |row, _| {
            0.8f64.mul_add((0.15f64 * usize_to_f64(row)).sin().abs(), 0.2)
        });
        let response = Mat::from_fn(n, 1, |row, _| (0.25f64 * usize_to_f64(row)).sin());

        let optimized = weighted_xtz(&x, &weights, &response);
        let diagonal = Mat::from_fn(
            n,
            n,
            |row, col| if row == col { weights[(row, 0)] } else { 0.0 },
        );
        let reference = x.transpose() * diagonal * response;

        for row in 0..p {
            assert!((optimized[(row, 0)] - reference[(row, 0)]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn logistic_irls_converges_on_synthetic_data() {
        let n = 64usize;
        let p = 3usize;
        let x = Mat::from_fn(n, p, |row, col| match col {
            0 => 1.0,
            1 => (0.17f64 * usize_to_f64(row)).sin(),
            _ => (0.11f64 * usize_to_f64(row)).cos(),
        });
        let y = Mat::from_fn(n, 1, |row, _| if row % 5 < 2 { 1.0 } else { 0.0 });
        let offset = (0..n)
            .map(|row| 0.05 * (0.09f64 * usize_to_f64(row)).sin())
            .collect::<Vec<_>>();

        let max_iter = 80usize;
        let tolerance = 1.0e-9;
        let (optimized_beta, optimized_iter) =
            fit_logistic_irls_with_offset(&x, &y, Some(&offset), max_iter, tolerance)
                .expect("optimized IRLS should converge");

        assert!(optimized_iter > 0);
        assert_eq!(optimized_beta.nrows(), p);
    }
}
