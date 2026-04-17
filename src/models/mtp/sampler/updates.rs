use rand::rngs::StdRng;

use super::math::{
    SpdInverseScratch, covariance_from_subject_effects, invert_matrix_with_jitter_and_scratch,
    linear_predictor, posterior_scale_matrix, random_walk_vector_into, sample_inverse_wishart,
    sample_standard_normal, should_accept, subject_effect_log_prior,
    usize_to_f64_reexport as usize_to_f64,
};
use super::{
    ChainState, FamilyRandomEffects, PosteriorCache, RowLikelihoodCache, SamplerContext,
    SubjectProposalEvaluation,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn update_alpha_block(
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
    std::mem::swap(&mut state.alpha, proposal_buffer);

    let candidate_binary_fixed = linear_predictor(&context.input.x_binary, &state.alpha);
    let Some(candidate_rows) = context.recompute_all_rows_from_offsets(
        state,
        &candidate_binary_fixed,
        &row_cache.mean_fixed,
        &row_cache.binary_offset,
        &row_cache.mean_offset,
    ) else {
        std::mem::swap(&mut state.alpha, proposal_buffer);
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
        row_cache.replace_all_rows_with_binary(candidate_rows, candidate_binary_fixed);
    } else {
        std::mem::swap(&mut state.alpha, proposal_buffer);
    }
    accepted
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update_beta_block(
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
    std::mem::swap(&mut state.beta, proposal_buffer);

    let candidate_mean_fixed = linear_predictor(&context.input.x_mean, &state.beta);
    let Some(candidate_rows) = context.recompute_all_rows_from_offsets(
        state,
        &row_cache.binary_fixed,
        &candidate_mean_fixed,
        &row_cache.binary_offset,
        &row_cache.mean_offset,
    ) else {
        std::mem::swap(&mut state.beta, proposal_buffer);
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
        row_cache.replace_all_rows_with_mean(candidate_rows, candidate_mean_fixed);
    } else {
        std::mem::swap(&mut state.beta, proposal_buffer);
    }
    accepted
}

pub(super) fn evaluate_subject_proposals(
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
pub(super) fn update_random_effects_block(
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
            let previous_effect = std::mem::replace(
                &mut state.subject_effects[subject_idx],
                proposal_buffer[subject_idx].clone(),
            );
            let rows = &context.subjects[subject_idx].rows;
            let likelihood_delta = row_cache.replace_rows(rows, &evaluation.candidate_row_values);
            row_cache.update_subject_offsets(
                &context.input.time,
                rows,
                context.structure,
                &previous_effect,
                &state.subject_effects[subject_idx],
            );
            posterior.log_likelihood += likelihood_delta;
            posterior.log_prior_random_effects += evaluation.prior_delta;
            accepted += 1;
        }
    }

    (accepted, proposed)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update_family_effects_block(
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
        let Some(candidate_rows) = context.family_candidate_rows_with_terms(
            row_cache,
            rows,
            current,
            candidate_effect,
            state.omega_sq,
            context.effective_kappa(state),
        ) else {
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
            row_cache.update_family_offsets(
                rows,
                candidate_effect[0] - current[0],
                candidate_effect[1] - current[1],
            );
            posterior.log_likelihood += likelihood_delta;
            posterior.log_prior_family += candidate_log_prior - current_log_prior;
            accepted += 1;
        }
    }

    (accepted, proposed)
}

pub(super) fn update_sigma_block(
    context: &SamplerContext<'_>,
    rng: &mut StdRng,
    state: &mut ChainState,
    posterior: &mut PosteriorCache,
    inverse_scratch: &mut SpdInverseScratch,
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

    let sampled_covariance =
        sample_inverse_wishart(rng, posterior_df, &posterior_scale, inverse_scratch)
            .unwrap_or_else(|_| {
                let mut fallback = covariance_from_subject_effects(&state.subject_effects);
                for diag in 0..dim {
                    fallback[(diag, diag)] = fallback[(diag, diag)].max(1.0e-6);
                }
                fallback
            });
    let sampled_precision =
        invert_matrix_with_jitter_and_scratch(&sampled_covariance, inverse_scratch);

    state.random_effects_cov = sampled_covariance;
    state.random_effects_precision = sampled_precision;
    posterior.log_prior_random_effects = SamplerContext::log_subject_random_effect_prior(state);
}

pub(super) fn update_kappa_block(
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

    let Some(candidate_rows) = context.recompute_all_rows_from_offsets(
        state,
        &row_cache.binary_fixed,
        &row_cache.mean_fixed,
        &row_cache.binary_offset,
        &row_cache.mean_offset,
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

pub(super) fn update_omega_block(
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

    let Some(candidate_rows) = context.recompute_all_rows_from_offsets(
        state,
        &row_cache.binary_fixed,
        &row_cache.mean_fixed,
        &row_cache.binary_offset,
        &row_cache.mean_offset,
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
