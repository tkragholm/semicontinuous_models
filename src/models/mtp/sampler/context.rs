use faer::Mat;

use crate::utils::{dot_row, quadratic_form_vec, solve_linear_system};

use super::likelihood::logistic_stable;
use super::{
    ChainState, FamilyRandomEffects, MIN_WEIGHT, MtpError, PositivePartDistribution,
    RE_UPDATE_TOLERANCE, RowLikelihoodCache, SUBJECT_NEWTON_MAX_ITERS, SamplerContext, SubjectRows,
    SubjectUpdateContext, likelihood, priors,
};

#[allow(clippy::wildcard_imports)]
use super::math::*;

impl SubjectUpdateContext<'_> {
    pub(super) fn optimize(
        &self,
        subject: &SubjectRows,
        current: &[f64],
    ) -> Result<Vec<f64>, MtpError> {
        let dim = current.len();
        let mut subject_effect = current.to_vec();
        let mut gradient_column = Mat::<f64>::zeros(dim, 1);

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

            for row in 0..dim {
                gradient_column[(row, 0)] = gradient[row];
            }
            let delta = solve_linear_system(&hessian, &gradient_column)
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
    pub(super) fn validate_state(&self, state: &ChainState) -> bool {
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

    pub(super) const fn effective_kappa(&self, state: &ChainState) -> f64 {
        match self.positive_part_distribution {
            PositivePartDistribution::LogSkewNormal => state.kappa,
            PositivePartDistribution::LogNormal => 0.0,
        }
    }

    #[cfg_attr(not(any(test, debug_assertions)), allow(dead_code))]
    pub(super) fn log_likelihood(&self, state: &ChainState) -> f64 {
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
    pub(super) fn row_log_likelihood(
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

    pub(super) fn row_family_effect(&self, state: &ChainState, row: usize) -> [f64; 2] {
        self.row_to_family.map_or([0.0, 0.0], |row_to_family| {
            state.family_effects[row_to_family[row]]
        })
    }

    pub(super) fn positive_row_log_likelihood(
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

    pub(super) fn row_log_likelihood_with_terms(
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

    pub(super) fn row_offsets(&self, state: &ChainState) -> (Vec<f64>, Vec<f64>) {
        let mut binary_offset = Vec::with_capacity(self.input.outcome.nrows());
        let mut mean_offset = Vec::with_capacity(self.input.outcome.nrows());
        for row in 0..self.input.outcome.nrows() {
            let subject_idx = self.row_to_subject[row];
            let subject_effect = &state.subject_effects[subject_idx];
            let family_effect = self.row_family_effect(state, row);
            binary_offset.push(
                random_binary_component(subject_effect, self.input.time[row], self.structure)
                    + family_effect[0],
            );
            mean_offset.push(
                random_mean_component(subject_effect, self.input.time[row], self.structure)
                    + family_effect[1],
            );
        }
        (binary_offset, mean_offset)
    }

    pub(super) fn recompute_all_rows_from_offsets(
        &self,
        state: &ChainState,
        binary_fixed: &[f64],
        mean_fixed: &[f64],
        binary_offset: &[f64],
        mean_offset: &[f64],
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
            let contribution = self.row_log_likelihood_with_terms(
                self.input.outcome[(row, 0)],
                binary_fixed[row] + binary_offset[row],
                mean_fixed[row] + mean_offset[row],
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

    pub(super) fn subject_candidate_rows_with_terms(
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
            let current_subject_effect = &state.subject_effects[subject_idx];
            let contribution = self.row_log_likelihood_with_terms(
                self.input.outcome[(*row, 0)],
                row_cache.binary_fixed[*row] + row_cache.binary_offset[*row]
                    - random_binary_component(
                        current_subject_effect,
                        self.input.time[*row],
                        self.structure,
                    )
                    + random_binary_component(
                        subject_effect,
                        self.input.time[*row],
                        self.structure,
                    ),
                row_cache.mean_fixed[*row] + row_cache.mean_offset[*row]
                    - random_mean_component(
                        current_subject_effect,
                        self.input.time[*row],
                        self.structure,
                    )
                    + random_mean_component(subject_effect, self.input.time[*row], self.structure),
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

    pub(super) fn family_candidate_rows_with_terms(
        &self,
        row_cache: &RowLikelihoodCache,
        rows: &[usize],
        current_family_effect: [f64; 2],
        family_effect: [f64; 2],
        omega_sq: f64,
        kappa: f64,
    ) -> Option<Vec<f64>> {
        if rows.is_empty() {
            return Some(Vec::new());
        }

        let omega = omega_sq.sqrt();
        let delta = kappa / kappa.mul_add(kappa, 1.0).sqrt();
        let log_phi_omega_delta =
            if self.positive_part_distribution == PositivePartDistribution::LogSkewNormal {
                likelihood::log_standard_normal_cdf(omega * delta)
            } else {
                0.0
            };

        let mut contributions = Vec::with_capacity(rows.len());
        for row in rows {
            let contribution = self.row_log_likelihood_with_terms(
                self.input.outcome[(*row, 0)],
                row_cache.binary_fixed[*row] + row_cache.binary_offset[*row]
                    - current_family_effect[0]
                    + family_effect[0],
                row_cache.mean_fixed[*row] + row_cache.mean_offset[*row] - current_family_effect[1]
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

    pub(super) fn log_alpha_prior(&self, alpha: &[f64]) -> f64 {
        alpha
            .iter()
            .map(|value| {
                priors::log_zero_mean_normal_density(*value, self.prior_config.alpha_variance)
            })
            .sum()
    }

    pub(super) fn log_beta_prior(&self, beta: &[f64]) -> f64 {
        beta.iter()
            .map(|value| {
                priors::log_zero_mean_normal_density(*value, self.prior_config.beta_variance)
            })
            .sum()
    }

    pub(super) fn log_kappa_prior(&self, kappa: f64) -> f64 {
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

    pub(super) fn log_omega_prior(&self, omega_sq: f64) -> f64 {
        priors::log_inverse_gamma_density(
            omega_sq,
            self.prior_config.omega_sq_shape,
            self.prior_config.omega_sq_scale,
        )
    }

    pub(super) fn family_effect_log_prior(&self, effect: [f64; 2]) -> f64 {
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

    pub(super) fn log_family_prior(&self, family_effects: &[[f64; 2]]) -> f64 {
        family_effects
            .iter()
            .copied()
            .map(|effect| self.family_effect_log_prior(effect))
            .sum()
    }

    pub(super) fn log_subject_random_effect_prior(state: &ChainState) -> f64 {
        (0..state.subject_effects.len())
            .map(|subject_idx| Self::subject_random_effect_log_prior(state, subject_idx))
            .sum()
    }

    pub(super) fn subject_random_effect_log_prior(state: &ChainState, subject_idx: usize) -> f64 {
        let effect = &state.subject_effects[subject_idx];
        -0.5 * quadratic_form_vec(effect, &state.random_effects_precision)
    }
}
