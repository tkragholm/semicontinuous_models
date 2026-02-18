//! Sampler entrypoints for correlated longitudinal MTP.

use std::collections::BTreeMap;

use faer::Mat;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::input::LongitudinalModelInput;
use crate::models::matrix_ops::{select_rows, select_values};
use crate::utils::{matrix_is_finite, max_abs_diff, solve_linear_system};

use super::diagnostics::summarize_multi_chain_convergence;
use super::input::{MtpPreparedInput, SubjectRows, prepare_input};
use super::likelihood::logistic_stable;
use super::posterior::{MtpPosteriorDraw, MtpPosteriorSamples, summarize_posterior};
use super::types::{
    FamilyRandomEffects, MtpAcceptanceRates, MtpError, MtpFitOptions, MtpModel,
    MtpMultiChainOptions, MtpMultiChainReport, MtpProposalTuning, MtpReport, MtpSamplerConfig,
    MtpSamplerDiagnostics, PositivePartDistribution, RandomEffectsStructure,
};
use super::{likelihood, priors};

const MIN_WEIGHT: f64 = 1.0e-8;
const RIDGE_L2: f64 = 1.0e-8;
const RE_UPDATE_TOLERANCE: f64 = 1.0e-6;
const OUTER_TOLERANCE: f64 = 1.0e-5;
const SUBJECT_NEWTON_MAX_ITERS: usize = 8;

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
        let row_log_likelihood = context
            .recompute_all_rows_with_terms(state, &binary_fixed, &mean_fixed)
            .ok_or(MtpError::NonConvergence)?;
        let total = row_log_likelihood.iter().sum::<f64>();

        if !total.is_finite() {
            return Err(MtpError::NonConvergence);
        }

        Ok(Self {
            row_log_likelihood,
            binary_fixed,
            mean_fixed,
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

    fn replace_all_rows(
        &mut self,
        row_log_likelihood: Vec<f64>,
        binary_fixed: Vec<f64>,
        mean_fixed: Vec<f64>,
    ) {
        self.total = row_log_likelihood.iter().sum();
        self.row_log_likelihood = row_log_likelihood;
        self.binary_fixed = binary_fixed;
        self.mean_fixed = mean_fixed;
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

    let model = MtpModel {
        n_binary_covariates: prepared.input.x_binary.ncols(),
        n_mean_covariates: prepared.input.x_mean.ncols(),
        random_effect_dimension: options.random_effects.random_effect_dimension(),
    };

    let report = MtpReport {
        diagnostics: MtpSamplerDiagnostics {
            iterations_completed: options.iterations,
            retained_draws,
            acceptance_rates: Some(sampling.acceptance_rates),
        },
        posterior_summary,
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

    let model = MtpModel {
        n_binary_covariates: prepared.input.x_binary.ncols(),
        n_mean_covariates: prepared.input.x_mean.ncols(),
        random_effect_dimension: options.random_effects.random_effect_dimension(),
    };

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
        chain_reports.push(MtpReport {
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

fn estimate_correlated_random_effects(
    prepared: &MtpPreparedInput<'_>,
    row_to_subject: &[usize],
    options: MtpFitOptions,
) -> Result<BaselineEstimate, MtpError> {
    let input = prepared.input;
    let structure = options.random_effects;
    let initial = initialize_estimation_state(input, options)?;
    let mut alpha = initial.alpha;
    let mut beta = initial.beta;

    let random_dim = structure.random_effect_dimension();
    let mut subject_effects = vec![vec![0.0; random_dim]; prepared.subjects.len()];
    let mut random_effects_cov = identity_matrix(random_dim);

    let mut omega_sq = residual_variance_with_subject_offsets(
        input,
        &initial.positive_indices,
        &beta,
        row_to_subject,
        &subject_effects,
        structure,
    );

    let outer_max_iters = options.iterations.clamp(1, 80);
    for _ in 0..outer_max_iters {
        let alpha_previous = alpha.clone();
        let beta_previous = beta.clone();
        let omega_previous = omega_sq;
        let subject_previous = subject_effects.clone();

        let binary_offsets = (0..input.outcome.nrows())
            .map(|row| {
                random_binary_component(
                    &subject_effects[row_to_subject[row]],
                    input.time[row],
                    structure,
                )
            })
            .collect::<Vec<_>>();

        let (alpha_next, _) = fit_logistic_irls_with_offset(
            &input.x_binary,
            &initial.binary_outcome,
            Some(&binary_offsets),
            40,
            1.0e-6,
        )?;
        alpha = column_to_vec(&alpha_next);

        let mean_offsets_positive = initial
            .positive_indices
            .iter()
            .map(|row| {
                random_mean_component(
                    &subject_effects[row_to_subject[*row]],
                    input.time[*row],
                    structure,
                )
            })
            .collect::<Vec<_>>();

        let adjusted_log_outcome =
            Mat::from_fn(initial.log_outcome_positive.nrows(), 1, |row, _| {
                initial.log_outcome_positive[(row, 0)] - mean_offsets_positive[row]
            });
        let beta_next = fit_linear_ols(&initial.x_mean_positive, &adjusted_log_outcome)?;
        beta = column_to_vec(&beta_next);

        omega_sq = residual_variance_with_subject_offsets(
            input,
            &initial.positive_indices,
            &beta,
            row_to_subject,
            &subject_effects,
            structure,
        );

        let prior_precision = invert_matrix_with_jitter(&random_effects_cov);
        let subject_context = SubjectUpdateContext {
            input,
            alpha: &alpha,
            beta: &beta,
            omega_sq,
            prior_precision: &prior_precision,
            structure,
        };
        for (subject_idx, subject) in prepared.subjects.iter().enumerate() {
            subject_effects[subject_idx] =
                subject_context.optimize(subject, &subject_effects[subject_idx])?;
        }

        random_effects_cov = covariance_from_subject_effects(&subject_effects);

        let max_change = max_slice_abs_diff(&alpha, &alpha_previous)
            .max(max_slice_abs_diff(&beta, &beta_previous))
            .max((omega_sq - omega_previous).abs())
            .max(max_subject_effect_change(
                &subject_effects,
                &subject_previous,
            ));

        if max_change < OUTER_TOLERANCE {
            break;
        }
    }

    Ok(BaselineEstimate {
        alpha,
        beta,
        subject_effects,
        random_effects_cov,
        kappa: 0.0,
        omega_sq,
        structure,
    })
}

fn initialize_estimation_state(
    input: &LongitudinalModelInput,
    options: MtpFitOptions,
) -> Result<InitialEstimationState, MtpError> {
    let binary_outcome = Mat::from_fn(input.outcome.nrows(), 1, |row, _| {
        if input.outcome[(row, 0)] > 0.0 {
            1.0
        } else {
            0.0
        }
    });

    let positive_indices: Vec<usize> = (0..input.outcome.nrows())
        .filter(|&row| input.outcome[(row, 0)] > 0.0)
        .collect();
    if positive_indices.is_empty() {
        return Err(MtpError::NoPositiveOutcomes);
    }

    let x_mean_positive = select_rows(&input.x_mean, &positive_indices);
    let outcome_positive = select_values(&input.outcome, &positive_indices);
    let log_outcome_positive = Mat::from_fn(outcome_positive.nrows(), 1, |row, _| {
        outcome_positive[(row, 0)].ln()
    });

    let (alpha_column, _) = fit_logistic_irls_with_offset(
        &input.x_binary,
        &binary_outcome,
        None,
        options.iterations.clamp(20, 50),
        1.0e-6,
    )?;
    let beta_column = fit_linear_ols(&x_mean_positive, &log_outcome_positive)?;

    Ok(InitialEstimationState {
        binary_outcome,
        positive_indices,
        x_mean_positive,
        log_outcome_positive,
        alpha: column_to_vec(&alpha_column),
        beta: column_to_vec(&beta_column),
    })
}

fn simulate_posterior_draws(
    request: &PosteriorSimulationRequest<'_>,
) -> Result<SamplingResult, MtpError> {
    let PosteriorSimulationRequest {
        input,
        subjects,
        row_to_subject,
        row_to_family,
        family_count,
        baseline,
        config,
        retained_draws,
    } = *request;

    let options = config.fit_options;
    let mut rng = StdRng::seed_from_u64(options.seed);
    let context = SamplerContext {
        input,
        subjects,
        row_to_subject,
        row_to_family,
        structure: baseline.structure,
        prior_config: config.prior_config,
        positive_part_distribution: config.positive_part_distribution,
        family_random_effects: config.family_random_effects,
    };
    let random_effects_precision = invert_matrix_with_jitter(&baseline.random_effects_cov);
    let mut state = ChainState {
        alpha: baseline.alpha.clone(),
        beta: baseline.beta.clone(),
        subject_effects: baseline.subject_effects.clone(),
        random_effects_cov: baseline.random_effects_cov.clone(),
        random_effects_precision,
        family_effects: vec![[0.0, 0.0]; family_count],
        kappa: baseline.kappa,
        omega_sq: baseline.omega_sq,
    };
    let proposal_scales =
        build_initial_proposal_scales(input, row_to_subject, baseline, config.proposal_tuning)?;
    run_mcmc_chain(
        &context,
        &mut rng,
        &mut state,
        proposal_scales,
        options,
        config.proposal_tuning,
        retained_draws,
    )
}

fn combine_posteriors(chains: &[MtpPosteriorSamples]) -> MtpPosteriorSamples {
    let total_draws = chains.iter().map(MtpPosteriorSamples::len).sum();
    let mut draws = Vec::with_capacity(total_draws);
    for chain in chains {
        draws.extend(chain.draws.iter().cloned());
    }
    MtpPosteriorSamples { draws }
}

fn build_initial_proposal_scales(
    input: &LongitudinalModelInput,
    row_to_subject: &[usize],
    baseline: &BaselineEstimate,
    tuning: MtpProposalTuning,
) -> Result<ProposalScales, MtpError> {
    Ok(ProposalScales {
        alpha: approximate_alpha_scales(input, row_to_subject, baseline)?
            .into_iter()
            .map(|value| value.max(tuning.min_draw_scale))
            .collect(),
        beta: approximate_beta_scales(input, baseline)?
            .into_iter()
            .map(|value| value.max(tuning.min_draw_scale))
            .collect(),
        random_effects: diagonal_from_matrix(&baseline.random_effects_cov)
            .into_iter()
            .map(|value| value.sqrt().max(tuning.min_draw_scale))
            .collect(),
        family_effects: [
            tuning.family_effect_draw_scale,
            tuning.family_effect_draw_scale,
        ],
        kappa: tuning.kappa_draw_scale,
        log_omega_sq: tuning.log_omega_draw_scale,
    })
}

#[allow(clippy::too_many_lines)]
fn run_mcmc_chain(
    context: &SamplerContext<'_>,
    rng: &mut StdRng,
    state: &mut ChainState,
    mut proposal_scales: ProposalScales,
    options: MtpFitOptions,
    tuning: MtpProposalTuning,
    retained_draws: usize,
) -> Result<SamplingResult, MtpError> {
    let mut row_cache = RowLikelihoodCache::initialize(context, state)?;
    let mut posterior =
        PosteriorCache::initialize_with_likelihood(context, state, row_cache.total)?;
    let family_rows = build_family_rows(context.row_to_family, state.family_effects.len());
    let mut counts = AcceptanceCounts::default();
    let mut draws = Vec::with_capacity(retained_draws);
    let mut buffers = SamplerBuffers {
        alpha_proposal: vec![0.0; state.alpha.len()],
        beta_proposal: vec![0.0; state.beta.len()],
        subject_proposals: state
            .subject_effects
            .iter()
            .map(|effect| vec![0.0; effect.len()])
            .collect(),
    };

    for iter in 0..options.iterations {
        counts.record_alpha(update_alpha_block(
            context,
            rng,
            state,
            &mut posterior,
            &mut row_cache,
            &mut buffers.alpha_proposal,
            &proposal_scales.alpha,
            tuning.min_draw_scale,
        ));
        counts.record_beta(update_beta_block(
            context,
            rng,
            state,
            &mut posterior,
            &mut row_cache,
            &mut buffers.beta_proposal,
            &proposal_scales.beta,
            tuning.min_draw_scale,
        ));
        let (accepted_random, proposed_random) = update_random_effects_block(
            context,
            rng,
            state,
            &mut posterior,
            &mut row_cache,
            &mut buffers.subject_proposals,
            &proposal_scales.random_effects,
            tuning.min_draw_scale,
        );
        counts.record_random_effects(accepted_random, proposed_random);
        let (accepted_family, proposed_family) = update_family_effects_block(
            context,
            rng,
            state,
            &mut posterior,
            &mut row_cache,
            &family_rows,
            proposal_scales.family_effects,
            tuning.min_draw_scale,
        );
        counts.record_family_effects(accepted_family, proposed_family);
        update_sigma_block(context, rng, state, &mut posterior);
        if context.positive_part_distribution == PositivePartDistribution::LogSkewNormal {
            counts.record_kappa(update_kappa_block(
                context,
                rng,
                state,
                &mut posterior,
                &mut row_cache,
                proposal_scales.kappa,
            ));
        }
        counts.record_omega(update_omega_block(
            context,
            rng,
            state,
            &mut posterior,
            &mut row_cache,
            proposal_scales.log_omega_sq,
        ));

        #[cfg(debug_assertions)]
        if iter.is_multiple_of(25) {
            let recomputed_likelihood = context.log_likelihood(state);
            debug_assert!((recomputed_likelihood - row_cache.total).abs() < 1.0e-8);
            debug_assert!((row_cache.total - posterior.log_likelihood).abs() < 1.0e-8);
            let recomputed_total =
                PosteriorCache::initialize_with_likelihood(context, state, row_cache.total)
                    .map_or(f64::NEG_INFINITY, PosteriorCache::total);
            debug_assert!((recomputed_total - posterior.total()).abs() < 1.0e-8);
        }

        if options.adapt_during_burn_in
            && iter < options.burn_in
            && (iter + 1).is_multiple_of(tuning.adaptation_interval)
        {
            adapt_proposal_scales(&mut proposal_scales, &counts, tuning);
        }

        if iter >= options.burn_in && (iter - options.burn_in).is_multiple_of(options.thin) {
            draws.push(MtpPosteriorDraw {
                alpha: state.alpha.clone(),
                beta: state.beta.clone(),
                kappa: state.kappa,
                omega_sq: state.omega_sq,
            });
        }
    }

    Ok(SamplingResult {
        samples: MtpPosteriorSamples { draws },
        acceptance_rates: counts.rates(),
    })
}

fn adapt_proposal_scales(
    proposal_scales: &mut ProposalScales,
    counts: &AcceptanceCounts,
    tuning: MtpProposalTuning,
) {
    let rates = counts.rates();
    adapt_vector_scale(&mut proposal_scales.alpha, rates.alpha, tuning);
    adapt_vector_scale(&mut proposal_scales.beta, rates.beta, tuning);
    adapt_vector_scale(
        &mut proposal_scales.random_effects,
        rates.random_effects,
        tuning,
    );
    for scale in &mut proposal_scales.family_effects {
        *scale = adapt_scalar_scale(*scale, rates.family_effects, tuning);
    }
    proposal_scales.kappa = adapt_scalar_scale(proposal_scales.kappa, rates.kappa, tuning);
    proposal_scales.log_omega_sq =
        adapt_scalar_scale(proposal_scales.log_omega_sq, rates.omega_sq, tuning);
}

impl SubjectUpdateContext<'_> {
    fn optimize(&self, subject: &SubjectRows, current: &[f64]) -> Result<Vec<f64>, MtpError> {
        let dim = current.len();
        let mut subject_effect = current.to_vec();

        for _ in 0..SUBJECT_NEWTON_MAX_ITERS {
            let mut gradient = vec![0.0; dim];
            let mut hessian = Mat::<f64>::zeros(dim, dim);

            for row in &subject.rows {
                let outcome = self.input.outcome[(*row, 0)];
                let is_positive = if outcome > 0.0 { 1.0 } else { 0.0 };
                let time = self.input.time[*row];

                let binary_basis = random_binary_basis(time, self.structure);
                let mean_basis = random_mean_basis(time, self.structure);

                let eta_binary = dot_row(&self.input.x_binary, *row, self.alpha)
                    + dot_basis_effect(&binary_basis, &subject_effect, dim);
                let probability = logistic_stable(eta_binary);
                let binary_weight = (probability * (1.0 - probability)).max(MIN_WEIGHT);

                add_scaled_basis(&mut gradient, &binary_basis, is_positive - probability, dim);
                add_scaled_outer_product(&mut hessian, &binary_basis, -binary_weight, dim);

                if outcome > 0.0 {
                    let eta_mean = dot_row(&self.input.x_mean, *row, self.beta)
                        + dot_basis_effect(&mean_basis, &subject_effect, dim);
                    let residual = outcome.ln() - eta_mean;
                    add_scaled_basis(&mut gradient, &mean_basis, residual / self.omega_sq, dim);
                    add_scaled_outer_product(
                        &mut hessian,
                        &mean_basis,
                        -(1.0 / self.omega_sq),
                        dim,
                    );
                }
            }

            for i in 0..dim {
                let penalty = (0..dim)
                    .map(|j| self.prior_precision[(i, j)] * subject_effect[j])
                    .sum::<f64>();
                gradient[i] -= penalty;
                for j in 0..dim {
                    hessian[(i, j)] -= self.prior_precision[(i, j)];
                }
                hessian[(i, i)] -= 1.0e-8;
            }

            let delta = solve_linear_system(&hessian, &vec_to_column(&gradient))
                .map_err(|_| MtpError::SolveFailed)?;
            let delta_vec = column_to_vec(&delta);

            for i in 0..dim {
                subject_effect[i] -= delta_vec[i];
            }

            let max_delta = delta_vec
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max);
            if max_delta < RE_UPDATE_TOLERANCE {
                break;
            }
        }

        Ok(subject_effect)
    }
}

impl SamplerContext<'_> {
    fn validate_state(&self, state: &ChainState) -> bool {
        if !(state.omega_sq.is_finite() && state.omega_sq > 0.0 && state.kappa.is_finite()) {
            return false;
        }
        if state.alpha.len() != self.input.x_binary.ncols()
            || state.beta.len() != self.input.x_mean.ncols()
            || state.subject_effects.len()
                != self.row_to_subject.iter().max().copied().unwrap_or(0) + 1
            || state.subject_effects.len() != self.subjects.len()
        {
            return false;
        }
        if self.family_random_effects != FamilyRandomEffects::Disabled {
            let Some(row_to_family) = self.row_to_family else {
                return false;
            };
            if state.family_effects.len() != row_to_family.iter().max().copied().unwrap_or(0) + 1 {
                return false;
            }
        }
        true
    }

    const fn effective_kappa(&self, state: &ChainState) -> f64 {
        match self.positive_part_distribution {
            PositivePartDistribution::LogSkewNormal => state.kappa,
            PositivePartDistribution::LogNormal => 0.0,
        }
    }

    #[cfg_attr(not(any(test, debug_assertions)), allow(dead_code))]
    fn log_likelihood(&self, state: &ChainState) -> f64 {
        let omega = state.omega_sq.sqrt();
        let kappa = self.effective_kappa(state);
        let delta = kappa / kappa.mul_add(kappa, 1.0).sqrt();
        let log_phi_omega_delta =
            if self.positive_part_distribution == PositivePartDistribution::LogSkewNormal {
                likelihood::log_standard_normal_cdf(omega * delta)
            } else {
                0.0
            };

        let mut sum = 0.0;
        for row in 0..self.input.outcome.nrows() {
            let contribution =
                self.row_log_likelihood(state, row, omega, kappa, log_phi_omega_delta);
            if !contribution.is_finite() {
                return f64::NEG_INFINITY;
            }
            sum += contribution;
        }
        sum
    }

    #[cfg_attr(not(any(test, debug_assertions)), allow(dead_code))]
    fn row_log_likelihood(
        &self,
        state: &ChainState,
        row: usize,
        omega: f64,
        kappa: f64,
        log_phi_omega_delta: f64,
    ) -> f64 {
        let subject_idx = self.row_to_subject[row];
        let subject_effect = &state.subject_effects[subject_idx];
        let family_effect = self.row_family_effect(state, row);
        let probability = likelihood::clamp_probability(logistic_stable(
            dot_row(&self.input.x_binary, row, &state.alpha)
                + random_binary_component(subject_effect, self.input.time[row], self.structure)
                + family_effect[0],
        ));

        let outcome = self.input.outcome[(row, 0)];
        if outcome > 0.0 {
            let marginal_log_mean = dot_row(&self.input.x_mean, row, &state.beta)
                + random_mean_component(subject_effect, self.input.time[row], self.structure)
                + family_effect[1];
            self.positive_row_log_likelihood(
                outcome,
                marginal_log_mean,
                probability,
                omega,
                kappa,
                log_phi_omega_delta,
            )
        } else {
            likelihood::zero_branch_log_likelihood(probability)
        }
    }

    fn row_family_effect(&self, state: &ChainState, row: usize) -> [f64; 2] {
        self.row_to_family.map_or([0.0, 0.0], |row_to_family| {
            state.family_effects[row_to_family[row]]
        })
    }

    fn positive_row_log_likelihood(
        &self,
        outcome: f64,
        marginal_log_mean: f64,
        probability: f64,
        omega: f64,
        kappa: f64,
        log_phi_omega_delta: f64,
    ) -> f64 {
        let omega_sq = omega * omega;
        match self.positive_part_distribution {
            PositivePartDistribution::LogSkewNormal => {
                let xi = 0.5f64.mul_add(
                    -omega_sq,
                    marginal_log_mean
                        - std::f64::consts::LN_2
                        - probability.ln()
                        - log_phi_omega_delta,
                );
                likelihood::positive_branch_log_likelihood(outcome, xi, omega, kappa, probability)
            }
            PositivePartDistribution::LogNormal => {
                let xi = (-0.5 * omega_sq).mul_add(1.0, marginal_log_mean - probability.ln());
                likelihood::positive_branch_log_likelihood_lognormal(
                    outcome,
                    xi,
                    omega,
                    probability,
                )
            }
        }
    }

    fn row_log_likelihood_with_terms(
        &self,
        outcome: f64,
        binary_linear: f64,
        mean_linear: f64,
        omega: f64,
        kappa: f64,
        log_phi_omega_delta: f64,
    ) -> f64 {
        let probability = likelihood::clamp_probability(logistic_stable(binary_linear));
        if outcome > 0.0 {
            self.positive_row_log_likelihood(
                outcome,
                mean_linear,
                probability,
                omega,
                kappa,
                log_phi_omega_delta,
            )
        } else {
            likelihood::zero_branch_log_likelihood(probability)
        }
    }

    fn recompute_all_rows_with_terms(
        &self,
        state: &ChainState,
        binary_fixed: &[f64],
        mean_fixed: &[f64],
    ) -> Option<Vec<f64>> {
        let omega = state.omega_sq.sqrt();
        let kappa = self.effective_kappa(state);
        let delta = kappa / kappa.mul_add(kappa, 1.0).sqrt();
        let log_phi_omega_delta =
            if self.positive_part_distribution == PositivePartDistribution::LogSkewNormal {
                likelihood::log_standard_normal_cdf(omega * delta)
            } else {
                0.0
            };

        let mut rows = Vec::with_capacity(self.input.outcome.nrows());
        for row in 0..self.input.outcome.nrows() {
            let subject_idx = self.row_to_subject[row];
            let subject_effect = &state.subject_effects[subject_idx];
            let family_effect = self.row_family_effect(state, row);
            let contribution = self.row_log_likelihood_with_terms(
                self.input.outcome[(row, 0)],
                binary_fixed[row]
                    + random_binary_component(subject_effect, self.input.time[row], self.structure)
                    + family_effect[0],
                mean_fixed[row]
                    + random_mean_component(subject_effect, self.input.time[row], self.structure)
                    + family_effect[1],
                omega,
                kappa,
                log_phi_omega_delta,
            );
            if !contribution.is_finite() {
                return None;
            }
            rows.push(contribution);
        }
        Some(rows)
    }

    fn subject_candidate_rows_with_terms(
        &self,
        state: &ChainState,
        row_cache: &RowLikelihoodCache,
        subject_idx: usize,
        subject_effect: &[f64],
    ) -> Option<Vec<f64>> {
        let omega = state.omega_sq.sqrt();
        let kappa = self.effective_kappa(state);
        let delta = kappa / kappa.mul_add(kappa, 1.0).sqrt();
        let log_phi_omega_delta =
            if self.positive_part_distribution == PositivePartDistribution::LogSkewNormal {
                likelihood::log_standard_normal_cdf(omega * delta)
            } else {
                0.0
            };

        let mut contributions = Vec::with_capacity(self.subjects[subject_idx].rows.len());
        for row in &self.subjects[subject_idx].rows {
            let family_effect = self.row_family_effect(state, *row);
            let contribution = self.row_log_likelihood_with_terms(
                self.input.outcome[(*row, 0)],
                row_cache.binary_fixed[*row]
                    + random_binary_component(
                        subject_effect,
                        self.input.time[*row],
                        self.structure,
                    )
                    + family_effect[0],
                row_cache.mean_fixed[*row]
                    + random_mean_component(subject_effect, self.input.time[*row], self.structure)
                    + family_effect[1],
                omega,
                kappa,
                log_phi_omega_delta,
            );
            if !contribution.is_finite() {
                return None;
            }
            contributions.push(contribution);
        }

        Some(contributions)
    }

    fn family_candidate_rows_with_terms(
        &self,
        state: &ChainState,
        row_cache: &RowLikelihoodCache,
        rows: &[usize],
        family_effect: [f64; 2],
    ) -> Option<Vec<f64>> {
        if rows.is_empty() {
            return Some(Vec::new());
        }

        let omega = state.omega_sq.sqrt();
        let kappa = self.effective_kappa(state);
        let delta = kappa / kappa.mul_add(kappa, 1.0).sqrt();
        let log_phi_omega_delta =
            if self.positive_part_distribution == PositivePartDistribution::LogSkewNormal {
                likelihood::log_standard_normal_cdf(omega * delta)
            } else {
                0.0
            };

        let mut contributions = Vec::with_capacity(rows.len());
        for row in rows {
            let subject_effect = &state.subject_effects[self.row_to_subject[*row]];
            let contribution = self.row_log_likelihood_with_terms(
                self.input.outcome[(*row, 0)],
                row_cache.binary_fixed[*row]
                    + random_binary_component(
                        subject_effect,
                        self.input.time[*row],
                        self.structure,
                    )
                    + family_effect[0],
                row_cache.mean_fixed[*row]
                    + random_mean_component(subject_effect, self.input.time[*row], self.structure)
                    + family_effect[1],
                omega,
                kappa,
                log_phi_omega_delta,
            );
            if !contribution.is_finite() {
                return None;
            }
            contributions.push(contribution);
        }

        Some(contributions)
    }

    fn log_alpha_prior(&self, alpha: &[f64]) -> f64 {
        alpha
            .iter()
            .map(|value| {
                priors::log_zero_mean_normal_density(*value, self.prior_config.alpha_variance)
            })
            .sum()
    }

    fn log_beta_prior(&self, beta: &[f64]) -> f64 {
        beta.iter()
            .map(|value| {
                priors::log_zero_mean_normal_density(*value, self.prior_config.beta_variance)
            })
            .sum()
    }

    fn log_kappa_prior(&self, kappa: f64) -> f64 {
        if self.positive_part_distribution == PositivePartDistribution::LogSkewNormal {
            priors::log_uniform_density(
                kappa,
                self.prior_config.kappa_lower,
                self.prior_config.kappa_upper,
            )
        } else {
            0.0
        }
    }

    fn log_omega_prior(&self, omega_sq: f64) -> f64 {
        priors::log_inverse_gamma_density(
            omega_sq,
            self.prior_config.omega_sq_shape,
            self.prior_config.omega_sq_scale,
        )
    }

    fn family_effect_log_prior(&self, effect: [f64; 2]) -> f64 {
        if self.family_random_effects == FamilyRandomEffects::Disabled {
            0.0
        } else {
            priors::log_zero_mean_normal_density(
                effect[0],
                self.prior_config.family_binary_variance,
            ) + priors::log_zero_mean_normal_density(
                effect[1],
                self.prior_config.family_mean_variance,
            )
        }
    }

    fn log_family_prior(&self, family_effects: &[[f64; 2]]) -> f64 {
        family_effects
            .iter()
            .copied()
            .map(|effect| self.family_effect_log_prior(effect))
            .sum()
    }

    fn log_subject_random_effect_prior(state: &ChainState) -> f64 {
        (0..state.subject_effects.len())
            .map(|subject_idx| Self::subject_random_effect_log_prior(state, subject_idx))
            .sum()
    }

    fn subject_random_effect_log_prior(state: &ChainState, subject_idx: usize) -> f64 {
        let effect = &state.subject_effects[subject_idx];
        let mut quadratic = 0.0;
        for row in 0..effect.len() {
            for col in 0..effect.len() {
                quadratic += effect[row] * state.random_effects_precision[(row, col)] * effect[col];
            }
        }
        -0.5 * quadratic
    }
}

#[allow(clippy::too_many_arguments)]
fn update_alpha_block(
    context: &SamplerContext<'_>,
    rng: &mut StdRng,
    state: &mut ChainState,
    posterior: &mut PosteriorCache,
    row_cache: &mut RowLikelihoodCache,
    proposal_buffer: &mut Vec<f64>,
    scales: &[f64],
    min_draw_scale: f64,
) -> bool {
    let current_total = posterior.total();
    random_walk_vector_into(proposal_buffer, &state.alpha, scales, rng, min_draw_scale);
    let previous = std::mem::replace(&mut state.alpha, proposal_buffer.clone());

    let candidate_binary_fixed = linear_predictor(&context.input.x_binary, &state.alpha);
    let Some(candidate_rows) = context.recompute_all_rows_with_terms(
        state,
        &candidate_binary_fixed,
        &row_cache.mean_fixed,
    ) else {
        state.alpha = previous;
        return false;
    };
    let candidate_log_likelihood = candidate_rows.iter().sum::<f64>();
    let candidate_alpha_prior = context.log_alpha_prior(&state.alpha);
    let candidate_total = current_total - posterior.log_likelihood - posterior.log_prior_alpha
        + candidate_log_likelihood
        + candidate_alpha_prior;
    let accepted = should_accept(candidate_total - current_total, rng);
    if accepted {
        posterior.log_likelihood = candidate_log_likelihood;
        posterior.log_prior_alpha = candidate_alpha_prior;
        row_cache.replace_all_rows(
            candidate_rows,
            candidate_binary_fixed,
            row_cache.mean_fixed.clone(),
        );
    } else {
        state.alpha = previous;
    }
    accepted
}

#[allow(clippy::too_many_arguments)]
fn update_beta_block(
    context: &SamplerContext<'_>,
    rng: &mut StdRng,
    state: &mut ChainState,
    posterior: &mut PosteriorCache,
    row_cache: &mut RowLikelihoodCache,
    proposal_buffer: &mut Vec<f64>,
    scales: &[f64],
    min_draw_scale: f64,
) -> bool {
    let current_total = posterior.total();
    random_walk_vector_into(proposal_buffer, &state.beta, scales, rng, min_draw_scale);
    let previous = std::mem::replace(&mut state.beta, proposal_buffer.clone());

    let candidate_mean_fixed = linear_predictor(&context.input.x_mean, &state.beta);
    let Some(candidate_rows) = context.recompute_all_rows_with_terms(
        state,
        &row_cache.binary_fixed,
        &candidate_mean_fixed,
    ) else {
        state.beta = previous;
        return false;
    };
    let candidate_log_likelihood = candidate_rows.iter().sum::<f64>();
    let candidate_beta_prior = context.log_beta_prior(&state.beta);
    let candidate_total = current_total - posterior.log_likelihood - posterior.log_prior_beta
        + candidate_log_likelihood
        + candidate_beta_prior;
    let accepted = should_accept(candidate_total - current_total, rng);
    if accepted {
        posterior.log_likelihood = candidate_log_likelihood;
        posterior.log_prior_beta = candidate_beta_prior;
        row_cache.replace_all_rows(
            candidate_rows,
            row_cache.binary_fixed.clone(),
            candidate_mean_fixed,
        );
    } else {
        state.beta = previous;
    }
    accepted
}

fn evaluate_subject_proposals(
    context: &SamplerContext<'_>,
    state: &ChainState,
    row_cache: &RowLikelihoodCache,
    proposals: &[Vec<f64>],
) -> Vec<Option<SubjectProposalEvaluation>> {
    let subject_count = state.subject_effects.len();
    if subject_count == 0 {
        return Vec::new();
    }

    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(subject_count);
    let chunk_size = subject_count.div_ceil(threads).max(1);
    let use_parallel = threads > 1 && subject_count >= 64;

    if !use_parallel {
        return (0..subject_count)
            .map(|subject_idx| {
                let rows = &context.subjects[subject_idx].rows;
                let current_sum = row_cache.sum_rows(rows);
                let current_prior = subject_effect_log_prior(
                    &state.subject_effects[subject_idx],
                    &state.random_effects_precision,
                );
                let candidate_rows = context.subject_candidate_rows_with_terms(
                    state,
                    row_cache,
                    subject_idx,
                    &proposals[subject_idx],
                )?;
                let candidate_sum = candidate_rows.iter().sum::<f64>();
                let candidate_prior = subject_effect_log_prior(
                    &proposals[subject_idx],
                    &state.random_effects_precision,
                );
                Some(SubjectProposalEvaluation {
                    candidate_row_values: candidate_rows,
                    current_sum,
                    candidate_sum,
                    prior_delta: candidate_prior - current_prior,
                })
            })
            .collect();
    }

    let mut evaluations = (0..subject_count)
        .map(|_| None)
        .collect::<Vec<Option<SubjectProposalEvaluation>>>();

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk_start in (0..subject_count).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(subject_count);
            handles.push(scope.spawn(move || {
                let mut local = Vec::with_capacity(chunk_end - chunk_start);
                for (subject_idx, proposal) in proposals
                    .iter()
                    .enumerate()
                    .take(chunk_end)
                    .skip(chunk_start)
                {
                    let rows = &context.subjects[subject_idx].rows;
                    let current_sum = row_cache.sum_rows(rows);
                    let current_prior = subject_effect_log_prior(
                        &state.subject_effects[subject_idx],
                        &state.random_effects_precision,
                    );
                    let Some(candidate_rows) = context.subject_candidate_rows_with_terms(
                        state,
                        row_cache,
                        subject_idx,
                        proposal,
                    ) else {
                        continue;
                    };
                    let candidate_sum = candidate_rows.iter().sum::<f64>();
                    let candidate_prior =
                        subject_effect_log_prior(proposal, &state.random_effects_precision);

                    local.push((
                        subject_idx,
                        SubjectProposalEvaluation {
                            candidate_row_values: candidate_rows,
                            current_sum,
                            candidate_sum,
                            prior_delta: candidate_prior - current_prior,
                        },
                    ));
                }
                local
            }));
        }

        for handle in handles {
            let chunk = handle.join().expect("subject proposal worker panicked");
            for (subject_idx, evaluation) in chunk {
                evaluations[subject_idx] = Some(evaluation);
            }
        }
    });

    evaluations
}

#[allow(clippy::too_many_arguments)]
fn update_random_effects_block(
    context: &SamplerContext<'_>,
    rng: &mut StdRng,
    state: &mut ChainState,
    posterior: &mut PosteriorCache,
    row_cache: &mut RowLikelihoodCache,
    proposal_buffer: &mut [Vec<f64>],
    scales: &[f64],
    min_draw_scale: f64,
) -> (usize, usize) {
    if state.subject_effects.is_empty() {
        return (0, 0);
    }

    for (index, effect) in state.subject_effects.iter().enumerate() {
        random_walk_vector_into(
            &mut proposal_buffer[index],
            effect,
            scales,
            rng,
            min_draw_scale,
        );
    }
    let evaluations = evaluate_subject_proposals(context, state, row_cache, proposal_buffer);

    let mut accepted = 0;
    let proposed = state.subject_effects.len();
    for (subject_idx, evaluation) in evaluations.into_iter().enumerate() {
        let Some(evaluation) = evaluation else {
            continue;
        };

        let delta = (evaluation.candidate_sum - evaluation.current_sum) + evaluation.prior_delta;
        if should_accept(delta, rng) {
            state.subject_effects[subject_idx].clone_from(&proposal_buffer[subject_idx]);
            let rows = &context.subjects[subject_idx].rows;
            let likelihood_delta = row_cache.replace_rows(rows, &evaluation.candidate_row_values);
            posterior.log_likelihood += likelihood_delta;
            posterior.log_prior_random_effects += evaluation.prior_delta;
            accepted += 1;
        }
    }

    (accepted, proposed)
}

#[allow(clippy::too_many_arguments)]
fn update_family_effects_block(
    context: &SamplerContext<'_>,
    rng: &mut StdRng,
    state: &mut ChainState,
    posterior: &mut PosteriorCache,
    row_cache: &mut RowLikelihoodCache,
    family_rows: &[Vec<usize>],
    scales: [f64; 2],
    min_draw_scale: f64,
) -> (usize, usize) {
    if context.family_random_effects == FamilyRandomEffects::Disabled
        || state.family_effects.is_empty()
    {
        return (0, 0);
    }

    let mut accepted = 0;
    let proposed = state.family_effects.len();

    for (family_idx, rows) in family_rows
        .iter()
        .enumerate()
        .take(state.family_effects.len())
    {
        let current = state.family_effects[family_idx];
        let current_sum = row_cache.sum_rows(rows);
        let current_log_prior = context.family_effect_log_prior(current);

        let candidate_effect = [
            scales[0]
                .max(min_draw_scale)
                .mul_add(sample_standard_normal(rng), current[0]),
            scales[1]
                .max(min_draw_scale)
                .mul_add(sample_standard_normal(rng), current[1]),
        ];
        let Some(candidate_rows) =
            context.family_candidate_rows_with_terms(state, row_cache, rows, candidate_effect)
        else {
            continue;
        };
        let candidate_sum = candidate_rows.iter().sum::<f64>();
        let candidate_log_prior = context.family_effect_log_prior(candidate_effect);

        if should_accept(
            (candidate_sum - current_sum) + (candidate_log_prior - current_log_prior),
            rng,
        ) {
            state.family_effects[family_idx] = candidate_effect;
            let likelihood_delta = row_cache.replace_rows(rows, &candidate_rows);
            posterior.log_likelihood += likelihood_delta;
            posterior.log_prior_family += candidate_log_prior - current_log_prior;
            accepted += 1;
        }
    }

    (accepted, proposed)
}

fn update_sigma_block(
    context: &SamplerContext<'_>,
    rng: &mut StdRng,
    state: &mut ChainState,
    posterior: &mut PosteriorCache,
) {
    let dim = state.random_effects_cov.ncols();
    let n_subjects = state.subject_effects.len();
    let prior_df = context
        .prior_config
        .random_effects_df
        .max(usize_to_f64(dim) + 1.0);
    let posterior_df = prior_df + usize_to_f64(n_subjects);
    let posterior_scale = posterior_scale_matrix(
        &state.subject_effects,
        dim,
        context.prior_config.random_effects_scale_diag,
    );

    let sampled_covariance = sample_inverse_wishart(rng, posterior_df, &posterior_scale)
        .unwrap_or_else(|_| {
            let mut fallback = covariance_from_subject_effects(&state.subject_effects);
            for diag in 0..dim {
                fallback[(diag, diag)] = fallback[(diag, diag)].max(1.0e-6);
            }
            fallback
        });
    let sampled_precision = invert_matrix_with_jitter(&sampled_covariance);

    state.random_effects_cov = sampled_covariance;
    state.random_effects_precision = sampled_precision;
    posterior.log_prior_random_effects = SamplerContext::log_subject_random_effect_prior(state);
}

fn update_kappa_block(
    context: &SamplerContext<'_>,
    rng: &mut StdRng,
    state: &mut ChainState,
    posterior: &mut PosteriorCache,
    row_cache: &mut RowLikelihoodCache,
    scale: f64,
) -> bool {
    let current_total = posterior.total();
    let previous = state.kappa;
    let kappa_lower = context.prior_config.kappa_lower;
    let kappa_upper = context.prior_config.kappa_upper;
    state.kappa = scale
        .mul_add(sample_standard_normal(rng), state.kappa)
        .clamp(kappa_lower, kappa_upper);

    let Some(candidate_rows) = context.recompute_all_rows_with_terms(
        state,
        &row_cache.binary_fixed,
        &row_cache.mean_fixed,
    ) else {
        state.kappa = previous;
        return false;
    };
    let candidate_log_likelihood = candidate_rows.iter().sum::<f64>();
    let candidate_kappa_prior = context.log_kappa_prior(state.kappa);
    let candidate_total = current_total - posterior.log_likelihood - posterior.log_prior_kappa
        + candidate_log_likelihood
        + candidate_kappa_prior;
    let accepted = should_accept(candidate_total - current_total, rng);
    if accepted {
        posterior.log_likelihood = candidate_log_likelihood;
        posterior.log_prior_kappa = candidate_kappa_prior;
        row_cache.row_log_likelihood = candidate_rows;
        row_cache.total = candidate_log_likelihood;
    } else {
        state.kappa = previous;
    }
    accepted
}

fn update_omega_block(
    context: &SamplerContext<'_>,
    rng: &mut StdRng,
    state: &mut ChainState,
    posterior: &mut PosteriorCache,
    row_cache: &mut RowLikelihoodCache,
    scale: f64,
) -> bool {
    let current_total = posterior.total();
    let previous = state.omega_sq;
    let proposed_log_omega = scale.mul_add(sample_standard_normal(rng), previous.ln());
    state.omega_sq = proposed_log_omega.exp().max(1.0e-8);

    let Some(candidate_rows) = context.recompute_all_rows_with_terms(
        state,
        &row_cache.binary_fixed,
        &row_cache.mean_fixed,
    ) else {
        state.omega_sq = previous;
        return false;
    };
    let candidate_log_likelihood = candidate_rows.iter().sum::<f64>();
    let candidate_omega_prior = context.log_omega_prior(state.omega_sq);
    let candidate_total = current_total - posterior.log_likelihood - posterior.log_prior_omega
        + candidate_log_likelihood
        + candidate_omega_prior;
    let log_acceptance = candidate_total + state.omega_sq.ln() - (current_total + previous.ln());
    let accepted = should_accept(log_acceptance, rng);
    if accepted {
        posterior.log_likelihood = candidate_log_likelihood;
        posterior.log_prior_omega = candidate_omega_prior;
        row_cache.row_log_likelihood = candidate_rows;
        row_cache.total = candidate_log_likelihood;
    } else {
        state.omega_sq = previous;
    }
    accepted
}

fn build_family_rows(row_to_family: Option<&[usize]>, family_count: usize) -> Vec<Vec<usize>> {
    let Some(row_to_family) = row_to_family else {
        return Vec::new();
    };
    if family_count == 0 {
        return Vec::new();
    }

    let mut family_rows = vec![Vec::new(); family_count];
    for (row, family_idx) in row_to_family.iter().copied().enumerate() {
        if family_idx < family_count {
            family_rows[family_idx].push(row);
        }
    }
    family_rows
}

fn random_walk_vector_into(
    output: &mut Vec<f64>,
    values: &[f64],
    scales: &[f64],
    rng: &mut StdRng,
    min_draw_scale: f64,
) {
    output.clear();
    output.extend(
        values
            .iter()
            .zip(scales.iter())
            .map(|(value, scale)| value + scale.max(min_draw_scale) * sample_standard_normal(rng)),
    );
}

fn should_accept(log_acceptance: f64, rng: &mut StdRng) -> bool {
    log_acceptance >= 0.0 || rng.random::<f64>().ln() < log_acceptance
}

fn acceptance_rate(accepted: usize, proposed: usize) -> f64 {
    if proposed == 0 {
        0.0
    } else {
        usize_to_f64(accepted) / usize_to_f64(proposed)
    }
}

fn adapt_vector_scale(scales: &mut [f64], acceptance: f64, tuning: MtpProposalTuning) {
    let factor = adaptation_factor(acceptance, tuning);
    for scale in scales {
        *scale = (*scale * factor).max(tuning.min_draw_scale);
    }
}

fn adapt_scalar_scale(scale: f64, acceptance: f64, tuning: MtpProposalTuning) -> f64 {
    (scale * adaptation_factor(acceptance, tuning)).max(tuning.min_draw_scale)
}

fn adaptation_factor(acceptance: f64, tuning: MtpProposalTuning) -> f64 {
    if acceptance < tuning.acceptance_target_low {
        tuning.scale_decrease_factor
    } else if acceptance > tuning.acceptance_target_high {
        tuning.scale_increase_factor
    } else {
        1.0
    }
}

fn fit_logistic_irls_with_offset(
    x: &Mat<f64>,
    y: &Mat<f64>,
    offset: Option<&[f64]>,
    max_iter: usize,
    tolerance: f64,
) -> Result<(Mat<f64>, usize), MtpError> {
    let mut beta = Mat::<f64>::zeros(x.ncols(), 1);

    for iteration in 0..max_iter {
        let eta = Mat::from_fn(x.nrows(), 1, |row, _| {
            let fixed = dot_row_mat(x, row, &beta);
            offset.map_or(fixed, |offset_values| fixed + offset_values[row])
        });

        let probability = Mat::from_fn(eta.nrows(), 1, |row, _| logistic_stable(eta[(row, 0)]));
        let weights = Mat::from_fn(probability.nrows(), 1, |row, _| {
            (probability[(row, 0)] * (1.0 - probability[(row, 0)])).max(MIN_WEIGHT)
        });

        let pseudo_response = Mat::from_fn(eta.nrows(), 1, |row, _| {
            eta[(row, 0)] + (y[(row, 0)] - probability[(row, 0)]) / weights[(row, 0)]
        });

        let adjusted_response = Mat::from_fn(eta.nrows(), 1, |row, _| {
            pseudo_response[(row, 0)] - offset.map_or(0.0, |offset_values| offset_values[row])
        });

        let beta_next = weighted_least_squares(x, &weights, &adjusted_response)?;
        if max_abs_diff(&beta_next, &beta) < tolerance {
            return Ok((beta_next, iteration + 1));
        }
        beta = beta_next;
    }

    Err(MtpError::NonConvergence)
}

fn fit_linear_ols(x: &Mat<f64>, y: &Mat<f64>) -> Result<Mat<f64>, MtpError> {
    let weights = Mat::from_fn(x.nrows(), 1, |_, _| 1.0);
    weighted_least_squares(x, &weights, y)
}

fn weighted_least_squares(
    x: &Mat<f64>,
    weights: &Mat<f64>,
    response: &Mat<f64>,
) -> Result<Mat<f64>, MtpError> {
    let mut information = weighted_xtx(x, weights);
    information += ridge_penalty(x.ncols(), RIDGE_L2);
    let weighted_response = weighted_xtz(x, weights, response);
    solve_linear_system(&information, &weighted_response).map_err(|_| MtpError::SolveFailed)
}

fn weighted_xtx(x: &Mat<f64>, weights: &Mat<f64>) -> Mat<f64> {
    let diagonal = Mat::from_fn(weights.nrows(), weights.nrows(), |row, col| {
        if row == col { weights[(row, 0)] } else { 0.0 }
    });
    x.transpose() * diagonal * x
}

fn weighted_xtz(x: &Mat<f64>, weights: &Mat<f64>, response: &Mat<f64>) -> Mat<f64> {
    let diagonal = Mat::from_fn(weights.nrows(), weights.nrows(), |row, col| {
        if row == col { weights[(row, 0)] } else { 0.0 }
    });
    x.transpose() * diagonal * response
}

fn ridge_penalty(ncols: usize, lambda: f64) -> Mat<f64> {
    Mat::from_fn(
        ncols,
        ncols,
        |row, col| {
            if row == col { lambda } else { 0.0 }
        },
    )
}

fn approximate_alpha_scales(
    input: &LongitudinalModelInput,
    row_to_subject: &[usize],
    baseline: &BaselineEstimate,
) -> Result<Vec<f64>, MtpError> {
    let alpha_column = vec_to_column(&baseline.alpha);
    let fixed_linear = &input.x_binary * &alpha_column;

    let eta = Mat::from_fn(input.x_binary.nrows(), 1, |row, _| {
        fixed_linear[(row, 0)]
            + random_binary_component(
                &baseline.subject_effects[row_to_subject[row]],
                input.time[row],
                baseline.structure,
            )
    });

    let probability = Mat::from_fn(eta.nrows(), 1, |row, _| logistic_stable(eta[(row, 0)]));
    let weights = Mat::from_fn(probability.nrows(), 1, |row, _| {
        (probability[(row, 0)] * (1.0 - probability[(row, 0)])).max(MIN_WEIGHT)
    });

    let mut information = weighted_xtx(&input.x_binary, &weights);
    let re_variance = random_block_variance(
        &baseline.random_effects_cov,
        baseline.structure,
        RandomBlock::Binary,
    );
    information += ridge_penalty(
        input.x_binary.ncols(),
        RIDGE_L2 + 1.0 / re_variance.max(1.0e-8),
    );
    diagonal_sqrt_inverse(&information)
}

fn approximate_beta_scales(
    input: &LongitudinalModelInput,
    baseline: &BaselineEstimate,
) -> Result<Vec<f64>, MtpError> {
    let positive_indices: Vec<usize> = (0..input.outcome.nrows())
        .filter(|&row| input.outcome[(row, 0)] > 0.0)
        .collect();
    if positive_indices.is_empty() {
        return Err(MtpError::NoPositiveOutcomes);
    }

    let x_mean_positive = select_rows(&input.x_mean, &positive_indices);
    let xtx = x_mean_positive.transpose() * &x_mean_positive;
    let mut information = Mat::from_fn(xtx.nrows(), xtx.ncols(), |row, col| {
        xtx[(row, col)] / baseline.omega_sq.max(1.0e-8)
    });

    let re_variance = random_block_variance(
        &baseline.random_effects_cov,
        baseline.structure,
        RandomBlock::Mean,
    );
    information += ridge_penalty(
        input.x_mean.ncols(),
        RIDGE_L2 + 1.0 / re_variance.max(1.0e-8),
    );

    let scales = diagonal_sqrt_inverse(&information)?;
    if scales.len() != baseline.beta.len() {
        return Err(MtpError::DesignCoefficientMismatch {
            design_cols: scales.len(),
            coef_len: baseline.beta.len(),
        });
    }

    Ok(scales)
}

fn diagonal_sqrt_inverse(matrix: &Mat<f64>) -> Result<Vec<f64>, MtpError> {
    let dim = matrix.ncols();
    let mut diagonal = Vec::with_capacity(dim);

    for col in 0..dim {
        let basis = Mat::from_fn(dim, 1, |row, _| if row == col { 1.0 } else { 0.0 });
        let solution = solve_linear_system(matrix, &basis).map_err(|_| MtpError::SolveFailed)?;
        diagonal.push(solution[(col, 0)].abs().sqrt().max(f64::MIN_POSITIVE));
    }

    Ok(diagonal)
}

fn residual_variance_with_subject_offsets(
    input: &LongitudinalModelInput,
    positive_indices: &[usize],
    beta: &[f64],
    row_to_subject: &[usize],
    subject_effects: &[Vec<f64>],
    structure: RandomEffectsStructure,
) -> f64 {
    let sum_squared = positive_indices
        .iter()
        .map(|row| {
            let row_idx = *row;
            let mean_value = dot_row(&input.x_mean, row_idx, beta)
                + random_mean_component(
                    &subject_effects[row_to_subject[row_idx]],
                    input.time[row_idx],
                    structure,
                );
            let residual = input.outcome[(row_idx, 0)].ln() - mean_value;
            residual * residual
        })
        .sum::<f64>();

    (sum_squared / usize_to_f64(positive_indices.len())).max(1.0e-8)
}

const fn random_binary_basis(time: f64, structure: RandomEffectsStructure) -> [f64; 4] {
    match structure {
        RandomEffectsStructure::InterceptsOnly => [1.0, 0.0, 0.0, 0.0],
        RandomEffectsStructure::InterceptsAndTimeSlopes => [1.0, time, 0.0, 0.0],
    }
}

const fn random_mean_basis(time: f64, structure: RandomEffectsStructure) -> [f64; 4] {
    match structure {
        RandomEffectsStructure::InterceptsOnly => [0.0, 1.0, 0.0, 0.0],
        RandomEffectsStructure::InterceptsAndTimeSlopes => [0.0, 0.0, 1.0, time],
    }
}

fn random_binary_component(effect: &[f64], time: f64, structure: RandomEffectsStructure) -> f64 {
    let basis = random_binary_basis(time, structure);
    dot_basis_effect(&basis, effect, effect.len())
}

fn random_mean_component(effect: &[f64], time: f64, structure: RandomEffectsStructure) -> f64 {
    let basis = random_mean_basis(time, structure);
    dot_basis_effect(&basis, effect, effect.len())
}

fn dot_basis_effect(basis: &[f64; 4], effect: &[f64], dim: usize) -> f64 {
    (0..dim).map(|idx| basis[idx] * effect[idx]).sum()
}

fn add_scaled_basis(gradient: &mut [f64], basis: &[f64; 4], scale: f64, dim: usize) {
    for idx in 0..dim {
        gradient[idx] += basis[idx] * scale;
    }
}

fn add_scaled_outer_product(hessian: &mut Mat<f64>, basis: &[f64; 4], scale: f64, dim: usize) {
    for row in 0..dim {
        for col in 0..dim {
            hessian[(row, col)] += scale * basis[row] * basis[col];
        }
    }
}

fn random_block_variance(
    covariance: &Mat<f64>,
    structure: RandomEffectsStructure,
    block: RandomBlock,
) -> f64 {
    let (start, len) = match (structure, block) {
        (RandomEffectsStructure::InterceptsOnly, RandomBlock::Binary) => (0, 1),
        (RandomEffectsStructure::InterceptsOnly, RandomBlock::Mean) => (1, 1),
        (RandomEffectsStructure::InterceptsAndTimeSlopes, RandomBlock::Binary) => (0, 2),
        (RandomEffectsStructure::InterceptsAndTimeSlopes, RandomBlock::Mean) => (2, 2),
    };

    let sum = (start..start + len)
        .map(|idx| covariance[(idx, idx)])
        .sum::<f64>();
    (sum / usize_to_f64(len)).max(1.0e-8)
}

fn build_row_to_subject_map(nrows: usize, subjects: &[SubjectRows]) -> Vec<usize> {
    let mut mapping = vec![usize::MAX; nrows];
    for (subject_idx, subject) in subjects.iter().enumerate() {
        for row in &subject.rows {
            mapping[*row] = subject_idx;
        }
    }
    debug_assert!(mapping.iter().all(|idx| *idx != usize::MAX));
    mapping
}

fn build_row_to_group_map(group_ids: &[u64]) -> (Vec<usize>, usize) {
    let mut mapping = vec![usize::MAX; group_ids.len()];
    let mut lookup: BTreeMap<u64, usize> = BTreeMap::new();
    let mut next_index = 0usize;

    for (row, group_id) in group_ids.iter().copied().enumerate() {
        let group_index = *lookup.entry(group_id).or_insert_with(|| {
            let index = next_index;
            next_index += 1;
            index
        });
        mapping[row] = group_index;
    }
    debug_assert!(mapping.iter().all(|idx| *idx != usize::MAX));
    (mapping, next_index)
}

fn covariance_from_subject_effects(subject_effects: &[Vec<f64>]) -> Mat<f64> {
    if subject_effects.is_empty() {
        return identity_matrix(2);
    }

    let dim = subject_effects[0].len();
    let n = usize_to_f64(subject_effects.len());

    let means = (0..dim)
        .map(|idx| {
            subject_effects
                .iter()
                .map(|effect| effect[idx])
                .sum::<f64>()
                / n
        })
        .collect::<Vec<_>>();

    let mut covariance = Mat::<f64>::zeros(dim, dim);
    for effect in subject_effects {
        for row in 0..dim {
            let centered_row = effect[row] - means[row];
            for col in 0..dim {
                let centered_col = effect[col] - means[col];
                covariance[(row, col)] += centered_row * centered_col;
            }
        }
    }

    for row in 0..dim {
        for col in 0..dim {
            covariance[(row, col)] /= n;
        }
    }

    for diag in 0..dim {
        covariance[(diag, diag)] = covariance[(diag, diag)].max(1.0e-6);
    }

    for row in 0..dim {
        for col in 0..dim {
            let symmetric = 0.5 * (covariance[(row, col)] + covariance[(col, row)]);
            covariance[(row, col)] = symmetric;
            covariance[(col, row)] = symmetric;
        }
    }

    covariance
}

fn posterior_scale_matrix(
    subject_effects: &[Vec<f64>],
    dim: usize,
    prior_scale_diag: f64,
) -> Mat<f64> {
    let mut scale = Mat::from_fn(
        dim,
        dim,
        |row, col| {
            if row == col { prior_scale_diag } else { 0.0 }
        },
    );
    for effect in subject_effects {
        for row in 0..dim {
            for col in 0..dim {
                scale[(row, col)] += effect[row] * effect[col];
            }
        }
    }
    for row in 0..dim {
        for col in 0..dim {
            let symmetric = 0.5 * (scale[(row, col)] + scale[(col, row)]);
            scale[(row, col)] = symmetric;
            scale[(col, row)] = symmetric;
        }
    }
    scale
}

fn sample_inverse_wishart(
    rng: &mut StdRng,
    df: f64,
    scale: &Mat<f64>,
) -> Result<Mat<f64>, MtpError> {
    let inv_scale = matrix_inverse(scale)?;
    let precision_sample = sample_wishart(rng, df, &inv_scale)?;
    matrix_inverse(&precision_sample)
}

fn sample_wishart(rng: &mut StdRng, df: f64, scale: &Mat<f64>) -> Result<Mat<f64>, MtpError> {
    let dim = scale.ncols();
    if dim == 0 || df <= usize_to_f64(dim.saturating_sub(1)) {
        return Err(MtpError::SolveFailed);
    }
    let chol = cholesky_lower(scale).ok_or(MtpError::SolveFailed)?;
    let mut bartlett = Mat::<f64>::zeros(dim, dim);
    for row in 0..dim {
        let dof = df - usize_to_f64(row);
        if dof <= 0.0 {
            return Err(MtpError::SolveFailed);
        }
        bartlett[(row, row)] = sample_chi_square(rng, dof).sqrt();
        for col in 0..row {
            bartlett[(row, col)] = sample_standard_normal(rng);
        }
    }
    let product = &chol * &bartlett;
    Ok(&product * product.transpose())
}

fn cholesky_lower(matrix: &Mat<f64>) -> Option<Mat<f64>> {
    let dim = matrix.ncols();
    if matrix.nrows() != dim {
        return None;
    }
    let mut lower = Mat::<f64>::zeros(dim, dim);
    for row in 0..dim {
        for col in 0..=row {
            let mut sum = matrix[(row, col)];
            for k in 0..col {
                sum -= lower[(row, k)] * lower[(col, k)];
            }
            if row == col {
                if sum <= 0.0 {
                    return None;
                }
                lower[(row, col)] = sum.sqrt();
            } else {
                let denom = lower[(col, col)];
                if denom <= 0.0 {
                    return None;
                }
                lower[(row, col)] = sum / denom;
            }
        }
    }
    Some(lower)
}

fn sample_chi_square(rng: &mut StdRng, dof: f64) -> f64 {
    sample_gamma(rng, 0.5 * dof, 2.0)
}

fn sample_gamma(rng: &mut StdRng, shape: f64, scale: f64) -> f64 {
    if !(shape > 0.0 && scale > 0.0) {
        return f64::NAN;
    }

    if shape < 1.0 {
        let u = (1.0_f64 - rng.random::<f64>()).max(f64::MIN_POSITIVE);
        return sample_gamma(rng, shape + 1.0, scale) * u.powf(1.0 / shape);
    }

    let shape_minus_third = shape - (1.0 / 3.0);
    let coeff = (1.0 / (9.0 * shape_minus_third)).sqrt();
    loop {
        let standard_normal = sample_standard_normal(rng);
        let one_plus_coeff_noise = coeff.mul_add(standard_normal, 1.0);
        if one_plus_coeff_noise <= 0.0 {
            continue;
        }
        let cubic_term = one_plus_coeff_noise * one_plus_coeff_noise * one_plus_coeff_noise;
        let uniform = rng.random::<f64>();
        if uniform
            < (0.0331 * standard_normal * standard_normal * standard_normal)
                .mul_add(-standard_normal, 1.0)
        {
            return scale * shape_minus_third * cubic_term;
        }
        if uniform.ln()
            < (0.5 * standard_normal).mul_add(
                standard_normal,
                shape_minus_third * (1.0 - cubic_term + cubic_term.ln()),
            )
        {
            return scale * shape_minus_third * cubic_term;
        }
    }
}

fn invert_matrix_with_jitter(matrix: &Mat<f64>) -> Mat<f64> {
    let dim = matrix.ncols();
    let mut jitter = 1.0e-8;

    for _ in 0..8 {
        let regularized = Mat::from_fn(dim, dim, |row, col| {
            if row == col {
                matrix[(row, col)] + jitter
            } else {
                matrix[(row, col)]
            }
        });

        if let Ok(inverse) = matrix_inverse(&regularized)
            && matrix_is_finite(&inverse)
        {
            return inverse;
        }
        jitter *= 10.0;
    }

    identity_matrix(dim)
}

fn matrix_inverse(matrix: &Mat<f64>) -> Result<Mat<f64>, MtpError> {
    let dim = matrix.ncols();
    let mut inverse = Mat::<f64>::zeros(dim, dim);

    for col in 0..dim {
        let basis = Mat::from_fn(dim, 1, |row, _| if row == col { 1.0 } else { 0.0 });
        let solution = solve_linear_system(matrix, &basis).map_err(|_| MtpError::SolveFailed)?;
        for row in 0..dim {
            inverse[(row, col)] = solution[(row, 0)];
        }
    }

    Ok(inverse)
}

fn max_subject_effect_change(current: &[Vec<f64>], previous: &[Vec<f64>]) -> f64 {
    current
        .iter()
        .zip(previous.iter())
        .map(|(now, before)| {
            now.iter()
                .zip(before.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

fn max_slice_abs_diff(current: &[f64], previous: &[f64]) -> f64 {
    current
        .iter()
        .zip(previous.iter())
        .map(|(now, before)| (now - before).abs())
        .fold(0.0, f64::max)
}

fn linear_predictor(design_matrix: &Mat<f64>, coefficients: &[f64]) -> Vec<f64> {
    let coefficients_column = vec_to_column(coefficients);
    let values = design_matrix * &coefficients_column;
    (0..values.nrows()).map(|row| values[(row, 0)]).collect()
}

fn subject_effect_log_prior(effect: &[f64], random_effects_precision: &Mat<f64>) -> f64 {
    let mut quadratic = 0.0;
    for row in 0..effect.len() {
        for col in 0..effect.len() {
            quadratic += effect[row] * random_effects_precision[(row, col)] * effect[col];
        }
    }
    -0.5 * quadratic
}

fn dot_row(matrix: &Mat<f64>, row: usize, coefficients: &[f64]) -> f64 {
    (0..matrix.ncols())
        .map(|col| matrix[(row, col)] * coefficients[col])
        .sum()
}

fn dot_row_mat(matrix: &Mat<f64>, row: usize, coefficients: &Mat<f64>) -> f64 {
    (0..matrix.ncols())
        .map(|col| matrix[(row, col)] * coefficients[(col, 0)])
        .sum()
}

fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = (1.0_f64 - rng.random::<f64>()).max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0_f64 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn diagonal_from_matrix(matrix: &Mat<f64>) -> Vec<f64> {
    let n = matrix.nrows().min(matrix.ncols());
    (0..n).map(|idx| matrix[(idx, idx)]).collect()
}

fn identity_matrix(dim: usize) -> Mat<f64> {
    Mat::from_fn(dim, dim, |row, col| if row == col { 1.0 } else { 0.0 })
}

fn vec_to_column(values: &[f64]) -> Mat<f64> {
    Mat::from_fn(values.len(), 1, |row, _| values[row])
}

fn column_to_vec(column: &Mat<f64>) -> Vec<f64> {
    (0..column.nrows()).map(|row| column[(row, 0)]).collect()
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use faer::Mat;

    use super::*;

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
                    (2.0 + subj_intercept + subj_slope * time).exp()
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
        let options = MtpFitOptions {
            iterations: 80,
            burn_in: 20,
            thin: 4,
            random_effects: RandomEffectsStructure::InterceptsOnly,
            ..MtpFitOptions::default()
        };

        let (model, report) = fit_mtp_input(&input, options).expect("fit should run");
        assert_eq!(model.random_effect_dimension, 2);
        assert_eq!(report.diagnostics.retained_draws, 15);
        assert!(report.posterior_summary.is_some());
        assert!(report.diagnostics.acceptance_rates.is_some());
    }

    #[test]
    fn fit_returns_model_and_report_for_slopes() {
        let input = basic_input();
        let options = MtpFitOptions {
            iterations: 80,
            burn_in: 20,
            thin: 4,
            random_effects: RandomEffectsStructure::InterceptsAndTimeSlopes,
            ..MtpFitOptions::default()
        };

        let (model, report) = fit_mtp_input(&input, options).expect("fit should run");
        assert_eq!(model.random_effect_dimension, 4);
        assert_eq!(report.diagnostics.retained_draws, 15);
        assert!(report.posterior_summary.is_some());
        assert!(report.diagnostics.acceptance_rates.is_some());
    }

    #[test]
    fn fit_with_posterior_returns_draws() {
        let input = basic_input();
        let options = MtpFitOptions {
            iterations: 60,
            burn_in: 20,
            thin: 2,
            random_effects: RandomEffectsStructure::InterceptsOnly,
            ..MtpFitOptions::default()
        };

        let (_, report, posterior) =
            fit_mtp_input_with_posterior(&input, options).expect("posterior should run");
        assert_eq!(posterior.len(), report.diagnostics.retained_draws);
    }

    #[test]
    fn multi_chain_fit_returns_convergence_summary() {
        let input = basic_input();
        let options = MtpFitOptions {
            iterations: 60,
            burn_in: 20,
            thin: 2,
            random_effects: RandomEffectsStructure::InterceptsOnly,
            ..MtpFitOptions::default()
        };
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
            fit_options: MtpFitOptions {
                iterations: 70,
                burn_in: 20,
                thin: 2,
                seed: 1_337,
                random_effects: RandomEffectsStructure::InterceptsOnly,
                ..MtpFitOptions::default()
            },
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
            fit_options: MtpFitOptions {
                iterations: 60,
                burn_in: 20,
                thin: 2,
                random_effects: RandomEffectsStructure::InterceptsOnly,
                ..MtpFitOptions::default()
            },
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
            fit_options: MtpFitOptions {
                iterations: 60,
                burn_in: 20,
                thin: 2,
                ..MtpFitOptions::default()
            },
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
            fit_options: MtpFitOptions {
                iterations: 60,
                burn_in: 20,
                thin: 2,
                ..MtpFitOptions::default()
            },
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
            MtpFitOptions {
                iterations: 40,
                burn_in: 10,
                thin: 2,
                random_effects: RandomEffectsStructure::InterceptsAndTimeSlopes,
                ..MtpFitOptions::default()
            },
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
            MtpFitOptions {
                iterations: 40,
                burn_in: 10,
                thin: 2,
                random_effects: RandomEffectsStructure::InterceptsOnly,
                ..MtpFitOptions::default()
            },
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
}
