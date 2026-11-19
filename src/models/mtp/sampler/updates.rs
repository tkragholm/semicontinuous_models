use epistat_rng::Stream;

use super::adaptive::BlockProposal;
use super::math::{
    SpdInverseScratch, correlated_random_walk_into, covariance_from_subject_effects,
    invert_matrix_with_jitter_and_scratch, linear_predictor_into, posterior_scale_matrix,
    proposal_factor, sample_inverse_wishart, sample_standard_normal, should_accept,
    subject_effect_log_prior, usize_to_f64_reexport as usize_to_f64,
};
use super::{
    ChainState, FamilyRandomEffects, PosteriorCache, RowLikelihoodCache, SamplerBuffers,
    SamplerContext, SubjectProposalEvaluation,
};

/// `log(1 - 2 delta^2 / pi)` with `delta = kappa / sqrt(1 + kappa^2)`: the log ratio of
/// the variance of a skew-normal with shape `kappa` to its squared scale `omega_sq`.
/// It is 0 at `kappa = 0` and lies above `log(1 - 2 / pi)`.
pub(super) fn log_skew_variance_factor(kappa: f64) -> f64 {
    let delta_sq = kappa * kappa / kappa.mul_add(kappa, 1.0);
    (-2.0 * delta_sq / std::f64::consts::PI).ln_1p()
}

/// Write the fixed-effect block `(alpha, beta, log(omega_sq))` of `state` into `theta`.
pub(super) fn fixed_effect_block_into(theta: &mut Vec<f64>, state: &ChainState) {
    theta.clear();
    theta.extend_from_slice(&state.alpha);
    theta.extend_from_slice(&state.beta);
    theta.push(state.omega_sq.ln());
}

/// Joint random-walk Metropolis update of `alpha`, `beta` and `log(omega_sq)`. The walk
/// on `log(omega_sq)` adds the Jacobian `log(omega_sq') - log(omega_sq)` to the
/// acceptance ratio.
pub(super) fn update_fixed_effects_block(
    context: &SamplerContext<'_>,
    rng: &mut Stream,
    state: &mut ChainState,
    posterior: &mut PosteriorCache,
    row_cache: &mut RowLikelihoodCache,
    buffers: &mut SamplerBuffers,
    proposal: &BlockProposal,
) -> bool {
    let n_alpha = state.alpha.len();
    let n_beta = state.beta.len();
    proposal.draw_step_into(&mut buffers.fixed_step, rng);
    let step = &buffers.fixed_step;

    buffers.alpha_proposal.clear();
    buffers.alpha_proposal.extend(
        state
            .alpha
            .iter()
            .zip(&step[..n_alpha])
            .map(|(value, delta)| value + delta),
    );
    buffers.beta_proposal.clear();
    buffers.beta_proposal.extend(
        state
            .beta
            .iter()
            .zip(&step[n_alpha..n_alpha + n_beta])
            .map(|(value, delta)| value + delta),
    );
    let previous_omega_sq = state.omega_sq;
    let candidate_omega_sq = previous_omega_sq * step[n_alpha + n_beta].exp();
    if !(candidate_omega_sq.is_finite() && candidate_omega_sq > 0.0) {
        return false;
    }

    linear_predictor_into(
        &mut buffers.binary_fixed_scratch,
        &context.input.x_binary,
        &buffers.alpha_proposal,
    );
    linear_predictor_into(
        &mut buffers.mean_fixed_scratch,
        &context.input.x_mean,
        &buffers.beta_proposal,
    );
    state.omega_sq = candidate_omega_sq;
    if !context.recompute_all_rows_from_offsets_into(
        &mut buffers.row_scratch,
        state,
        &buffers.binary_fixed_scratch,
        &buffers.mean_fixed_scratch,
        &row_cache.binary_offset,
        &row_cache.mean_offset,
    ) {
        state.omega_sq = previous_omega_sq;
        return false;
    }

    let candidate_log_likelihood = buffers.row_scratch.iter().sum::<f64>();
    let candidate_alpha_prior = context.log_alpha_prior(&buffers.alpha_proposal);
    let candidate_beta_prior = context.log_beta_prior(&buffers.beta_proposal);
    let candidate_omega_prior = context.log_omega_prior(candidate_omega_sq);
    let log_acceptance = (candidate_log_likelihood - posterior.log_likelihood)
        + (candidate_alpha_prior - posterior.log_prior_alpha)
        + (candidate_beta_prior - posterior.log_prior_beta)
        + (candidate_omega_prior - posterior.log_prior_omega)
        + (candidate_omega_sq.ln() - previous_omega_sq.ln());

    let accepted = should_accept(log_acceptance, rng);
    if accepted {
        std::mem::swap(&mut state.alpha, &mut buffers.alpha_proposal);
        std::mem::swap(&mut state.beta, &mut buffers.beta_proposal);
        posterior.log_likelihood = candidate_log_likelihood;
        posterior.log_prior_alpha = candidate_alpha_prior;
        posterior.log_prior_beta = candidate_beta_prior;
        posterior.log_prior_omega = candidate_omega_prior;
        row_cache.accept_fixed_effects(buffers, candidate_log_likelihood);
    } else {
        state.omega_sq = previous_omega_sq;
    }
    accepted
}

/// Random-walk Metropolis update of `kappa` that holds the variance of `log(y)` given
/// `y > 0`, `omega_sq * (1 - 2 delta^2 / pi)`, fixed and moves `omega_sq` with it.
///
/// The data identify that variance well, while `kappa` is weakly identified near 0,
/// where the information matrix of the skew-normal is singular (Azzalini 1985). At a
/// fixed variance `omega_sq` grows with `|kappa|`, so a walk in `kappa` alone at fixed
/// `omega_sq` is held to small steps, and this walk is not. The step is a walk in the
/// `kappa` coordinate of `(log variance, kappa)`, the variance coordinate of Azzalini's
/// centred parameterisation, in which the target density carries the Jacobian
/// `omega_sq`, so `log(omega_sq') - log(omega_sq)` enters the acceptance ratio. A
/// `kappa` outside its uniform prior has zero prior density and is rejected.
pub(super) fn update_kappa_block(
    context: &SamplerContext<'_>,
    rng: &mut Stream,
    state: &mut ChainState,
    posterior: &mut PosteriorCache,
    row_cache: &mut RowLikelihoodCache,
    row_scratch: &mut Vec<f64>,
    scale: f64,
) -> bool {
    let previous_kappa = state.kappa;
    let previous_omega_sq = state.omega_sq;
    let candidate_kappa = scale.mul_add(sample_standard_normal(rng), previous_kappa);
    let candidate_kappa_prior = context.log_kappa_prior(candidate_kappa);
    let candidate_omega_sq = previous_omega_sq
        * (log_skew_variance_factor(previous_kappa) - log_skew_variance_factor(candidate_kappa))
            .exp();
    if !candidate_kappa_prior.is_finite()
        || !(candidate_omega_sq.is_finite() && candidate_omega_sq > 0.0)
    {
        return false;
    }

    state.kappa = candidate_kappa;
    state.omega_sq = candidate_omega_sq;
    if !context.recompute_all_rows_from_offsets_into(
        row_scratch,
        state,
        &row_cache.binary_fixed,
        &row_cache.mean_fixed,
        &row_cache.binary_offset,
        &row_cache.mean_offset,
    ) {
        state.kappa = previous_kappa;
        state.omega_sq = previous_omega_sq;
        return false;
    }

    let candidate_log_likelihood = row_scratch.iter().sum::<f64>();
    let candidate_omega_prior = context.log_omega_prior(candidate_omega_sq);
    let log_acceptance = (candidate_log_likelihood - posterior.log_likelihood)
        + (candidate_kappa_prior - posterior.log_prior_kappa)
        + (candidate_omega_prior - posterior.log_prior_omega)
        + (candidate_omega_sq.ln() - previous_omega_sq.ln());

    let accepted = should_accept(log_acceptance, rng);
    if accepted {
        posterior.log_likelihood = candidate_log_likelihood;
        posterior.log_prior_kappa = candidate_kappa_prior;
        posterior.log_prior_omega = candidate_omega_prior;
        row_cache.accept_rows(row_scratch, candidate_log_likelihood);
    } else {
        state.kappa = previous_kappa;
        state.omega_sq = previous_omega_sq;
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
    rng: &mut Stream,
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

    let factor = proposal_factor(&state.random_effects_cov);
    for (index, effect) in state.subject_effects.iter().enumerate() {
        correlated_random_walk_into(
            &mut proposal_buffer[index],
            effect,
            &factor,
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
    rng: &mut Stream,
    state: &mut ChainState,
    posterior: &mut PosteriorCache,
    row_cache: &mut RowLikelihoodCache,
    family_row_scratch: &mut Vec<f64>,
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
        if !context.family_candidate_rows_into(
            family_row_scratch,
            row_cache,
            rows,
            current,
            candidate_effect,
            state.omega_sq,
            context.effective_kappa(state),
        ) {
            continue;
        }
        let candidate_sum = family_row_scratch.iter().sum::<f64>();
        let candidate_log_prior = context.family_effect_log_prior(candidate_effect);

        if should_accept(
            (candidate_sum - current_sum) + (candidate_log_prior - current_log_prior),
            rng,
        ) {
            state.family_effects[family_idx] = candidate_effect;
            let likelihood_delta = row_cache.replace_rows(rows, family_row_scratch);
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
    rng: &mut Stream,
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
