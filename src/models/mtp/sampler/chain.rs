use faer::Mat;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::input::LongitudinalModelInput;
use crate::models::matrix_ops::{select_rows, select_values};
use crate::models::mtp::input::MtpPreparedInput;
use crate::models::mtp::posterior::{MtpPosteriorDraw, MtpPosteriorSamples};
use crate::models::mtp::types::{
    MtpError, MtpFitOptions, MtpProposalTuning, PositivePartDistribution,
};
use crate::utils::to_binary_outcome;

use super::{
    AcceptanceCounts, BaselineEstimate, ChainState, InitialEstimationState, OUTER_TOLERANCE,
    PosteriorCache, PosteriorSimulationRequest, ProposalScales, RowLikelihoodCache, SamplerBuffers,
    SamplerContext, SamplingResult, SubjectUpdateContext,
};

#[allow(clippy::wildcard_imports)]
use super::math::*;
#[allow(clippy::wildcard_imports)]
use super::updates::*;

pub(super) fn estimate_correlated_random_effects(
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
    let mut inverse_scratch = SpdInverseScratch::new(random_dim);

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

        let prior_precision =
            invert_matrix_with_jitter_and_scratch(&random_effects_cov, &mut inverse_scratch);
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

pub(super) fn initialize_estimation_state(
    input: &LongitudinalModelInput,
    options: MtpFitOptions,
) -> Result<InitialEstimationState, MtpError> {
    let binary_outcome = to_binary_outcome(&input.outcome);

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

pub(super) fn simulate_posterior_draws(
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
    let mut inverse_scratch = SpdInverseScratch::new(baseline.random_effects_cov.ncols());
    let random_effects_precision =
        invert_matrix_with_jitter_and_scratch(&baseline.random_effects_cov, &mut inverse_scratch);
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

pub(super) fn combine_posteriors(chains: &[MtpPosteriorSamples]) -> MtpPosteriorSamples {
    let total_draws = chains.iter().map(MtpPosteriorSamples::len).sum();
    let mut draws = Vec::with_capacity(total_draws);
    for chain in chains {
        draws.extend(chain.draws.iter().cloned());
    }
    MtpPosteriorSamples { draws }
}

pub(super) fn build_initial_proposal_scales(
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
pub(super) fn run_mcmc_chain(
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
    let mut inverse_scratch = SpdInverseScratch::new(state.random_effects_cov.ncols());

    for iter in 0usize..options.iterations {
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
        update_sigma_block(context, rng, state, &mut posterior, &mut inverse_scratch);
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

pub(super) fn adapt_proposal_scales(
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
