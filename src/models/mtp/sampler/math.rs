use std::collections::BTreeMap;

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par, Side, Spec};
use rand::RngExt;
use rand::rngs::StdRng;

use crate::input::LongitudinalModelInput;
use crate::models::matrix_ops::select_rows;
use crate::models::mtp::input::SubjectRows;
use crate::models::mtp::likelihood::logistic_stable;
use crate::models::mtp::types::{MtpError, MtpProposalTuning, RandomEffectsStructure};
pub(super) use crate::utils::usize_to_f64 as usize_to_f64_reexport;
use crate::utils::{
    add_ridge_to_diagonal, dot_row, dot_row_mat, matrix_is_finite, max_abs_diff,
    quadratic_form_vec, solve_linear_system, usize_to_f64, weighted_xtx, weighted_xtz_with_buffer,
};

use super::{BaselineEstimate, MIN_WEIGHT, RIDGE_L2, RandomBlock};

pub(super) fn build_family_rows(
    row_to_family: Option<&[usize]>,
    family_count: usize,
) -> Vec<Vec<usize>> {
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

pub(super) fn random_walk_vector_into(
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

pub(super) fn should_accept(log_acceptance: f64, rng: &mut StdRng) -> bool {
    log_acceptance >= 0.0 || rng.random::<f64>().ln() < log_acceptance
}

pub(super) fn adapt_vector_scale(scales: &mut [f64], acceptance: f64, tuning: MtpProposalTuning) {
    let factor = adaptation_factor(acceptance, tuning);
    for scale in scales {
        *scale = (*scale * factor).max(tuning.min_draw_scale);
    }
}

pub(super) fn adapt_scalar_scale(scale: f64, acceptance: f64, tuning: MtpProposalTuning) -> f64 {
    (scale * adaptation_factor(acceptance, tuning)).max(tuning.min_draw_scale)
}

pub(super) fn adaptation_factor(acceptance: f64, tuning: MtpProposalTuning) -> f64 {
    if acceptance < tuning.acceptance_target_low {
        tuning.scale_decrease_factor
    } else if acceptance > tuning.acceptance_target_high {
        tuning.scale_increase_factor
    } else {
        1.0
    }
}

pub(super) fn fit_logistic_irls_with_offset(
    x: &Mat<f64>,
    y: &Mat<f64>,
    offset: Option<&[f64]>,
    max_iter: usize,
    tolerance: f64,
) -> Result<(Mat<f64>, usize), MtpError> {
    let mut beta = Mat::<f64>::zeros(x.ncols(), 1);
    let mut weighted_xtz_buffer = Vec::new();

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

        let beta_next =
            weighted_least_squares(x, &weights, &adjusted_response, &mut weighted_xtz_buffer)?;
        if max_abs_diff(&beta_next, &beta) < tolerance {
            return Ok((beta_next, iteration + 1));
        }
        beta = beta_next;
    }

    Err(MtpError::NonConvergence)
}

pub(super) fn fit_linear_ols(x: &Mat<f64>, y: &Mat<f64>) -> Result<Mat<f64>, MtpError> {
    let weights = Mat::from_fn(x.nrows(), 1, |_, _| 1.0);
    let mut weighted_xtz_buffer = Vec::new();
    weighted_least_squares(x, &weights, y, &mut weighted_xtz_buffer)
}

pub(super) fn weighted_least_squares(
    x: &Mat<f64>,
    weights: &Mat<f64>,
    response: &Mat<f64>,
    weighted_buffer: &mut Vec<f64>,
) -> Result<Mat<f64>, MtpError> {
    let mut information = weighted_xtx(x, weights);
    add_ridge_to_diagonal(&mut information, RIDGE_L2, false);
    let weighted_response = weighted_xtz_with_buffer(x, weights, response, weighted_buffer);
    solve_linear_system(&information, &weighted_response).map_err(|_| MtpError::SolveFailed)
}

pub(super) fn approximate_alpha_scales(
    input: &LongitudinalModelInput,
    row_to_subject: &[usize],
    baseline: &BaselineEstimate,
) -> Result<Vec<f64>, MtpError> {
    let alpha_column = MatRef::from_column_major_slice(&baseline.alpha, baseline.alpha.len(), 1);
    let fixed_linear = &input.x_binary * alpha_column;

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
    add_ridge_to_diagonal(
        &mut information,
        RIDGE_L2 + 1.0 / re_variance.max(1.0e-8),
        false,
    );
    diagonal_sqrt_inverse(&information)
}

pub(super) fn approximate_beta_scales(
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
    add_ridge_to_diagonal(
        &mut information,
        RIDGE_L2 + 1.0 / re_variance.max(1.0e-8),
        false,
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

pub(super) fn diagonal_sqrt_inverse(matrix: &Mat<f64>) -> Result<Vec<f64>, MtpError> {
    let dim = matrix.ncols();
    let identity = Mat::<f64>::identity(dim, dim);
    let inverse = solve_linear_system(matrix, &identity).map_err(|_| MtpError::SolveFailed)?;
    let diagonal = (0..dim)
        .map(|idx| inverse[(idx, idx)].abs().sqrt().max(f64::MIN_POSITIVE))
        .collect();

    Ok(diagonal)
}

pub(super) fn residual_variance_with_subject_offsets(
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

pub(super) const fn random_binary_basis(time: f64, structure: RandomEffectsStructure) -> [f64; 4] {
    match structure {
        RandomEffectsStructure::InterceptsOnly => [1.0, 0.0, 0.0, 0.0],
        RandomEffectsStructure::InterceptsAndTimeSlopes => [1.0, time, 0.0, 0.0],
    }
}

pub(super) const fn random_mean_basis(time: f64, structure: RandomEffectsStructure) -> [f64; 4] {
    match structure {
        RandomEffectsStructure::InterceptsOnly => [0.0, 1.0, 0.0, 0.0],
        RandomEffectsStructure::InterceptsAndTimeSlopes => [0.0, 0.0, 1.0, time],
    }
}

pub(super) fn random_binary_component(
    effect: &[f64],
    time: f64,
    structure: RandomEffectsStructure,
) -> f64 {
    let basis = random_binary_basis(time, structure);
    dot_basis_effect(&basis, effect, effect.len())
}

pub(super) fn random_mean_component(
    effect: &[f64],
    time: f64,
    structure: RandomEffectsStructure,
) -> f64 {
    let basis = random_mean_basis(time, structure);
    dot_basis_effect(&basis, effect, effect.len())
}

pub(super) fn dot_basis_effect(basis: &[f64; 4], effect: &[f64], dim: usize) -> f64 {
    (0..dim).map(|idx| basis[idx] * effect[idx]).sum()
}

pub(super) fn add_scaled_basis(gradient: &mut [f64], basis: &[f64; 4], scale: f64, dim: usize) {
    for idx in 0..dim {
        gradient[idx] = basis[idx].mul_add(scale, gradient[idx]);
    }
}

pub(super) fn add_scaled_outer_product(
    hessian: &mut Mat<f64>,
    basis: &[f64; 4],
    scale: f64,
    dim: usize,
) {
    for row in 0..dim {
        for col in 0..dim {
            hessian[(row, col)] = (scale * basis[row]).mul_add(basis[col], hessian[(row, col)]);
        }
    }
}

pub(super) fn random_block_variance(
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

pub(super) fn build_row_to_subject_map(nrows: usize, subjects: &[SubjectRows]) -> Vec<usize> {
    let mut mapping = vec![usize::MAX; nrows];
    for (subject_idx, subject) in subjects.iter().enumerate() {
        for row in &subject.rows {
            mapping[*row] = subject_idx;
        }
    }
    debug_assert!(mapping.iter().all(|idx| *idx != usize::MAX));
    mapping
}

pub(super) fn build_row_to_group_map(group_ids: &[u64]) -> (Vec<usize>, usize) {
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

pub(super) fn covariance_from_subject_effects(subject_effects: &[Vec<f64>]) -> Mat<f64> {
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
                covariance[(row, col)] = centered_row.mul_add(centered_col, covariance[(row, col)]);
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

pub(super) fn posterior_scale_matrix(
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
                scale[(row, col)] = effect[row].mul_add(effect[col], scale[(row, col)]);
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

pub(super) fn sample_inverse_wishart(
    rng: &mut StdRng,
    df: f64,
    scale: &Mat<f64>,
    inverse_scratch: &mut SpdInverseScratch,
) -> Result<Mat<f64>, MtpError> {
    let inv_scale =
        matrix_inverse_with_scratch(scale, inverse_scratch).or_else(|_| matrix_inverse(scale))?;
    let precision_sample = sample_wishart(rng, df, &inv_scale)?;
    matrix_inverse_with_scratch(&precision_sample, inverse_scratch)
        .or_else(|_| matrix_inverse(&precision_sample))
}

pub(super) fn sample_wishart(
    rng: &mut StdRng,
    df: f64,
    scale: &Mat<f64>,
) -> Result<Mat<f64>, MtpError> {
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
    let mut product = Mat::<f64>::zeros(dim, dim);
    matmul(
        product.as_mut(),
        Accum::Replace,
        chol.as_ref(),
        bartlett.as_ref(),
        1.0,
        Par::Seq,
    );
    let mut wishart = Mat::<f64>::zeros(dim, dim);
    matmul(
        wishart.as_mut(),
        Accum::Replace,
        product.as_ref(),
        product.transpose(),
        1.0,
        Par::Seq,
    );
    Ok(wishart)
}

pub(super) fn cholesky_lower(matrix: &Mat<f64>) -> Option<Mat<f64>> {
    let dim = matrix.ncols();
    if matrix.nrows() != dim {
        return None;
    }
    let llt = matrix.as_ref().llt(Side::Lower).ok()?;
    let lower_factor = llt.L();
    Some(Mat::from_fn(dim, dim, |row, col| {
        if row >= col {
            lower_factor[(row, col)]
        } else {
            0.0
        }
    }))
}

pub(super) fn sample_chi_square(rng: &mut StdRng, dof: f64) -> f64 {
    sample_gamma(rng, 0.5 * dof, 2.0)
}

pub(super) fn sample_gamma(rng: &mut StdRng, shape: f64, scale: f64) -> f64 {
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

#[cfg(test)]
pub(super) fn invert_matrix_with_jitter(matrix: &Mat<f64>) -> Mat<f64> {
    let mut scratch = SpdInverseScratch::new(matrix.ncols());
    invert_matrix_with_jitter_and_scratch(matrix, &mut scratch)
}

pub(super) fn invert_matrix_with_jitter_and_scratch(
    matrix: &Mat<f64>,
    inverse_scratch: &mut SpdInverseScratch,
) -> Mat<f64> {
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

        if let Ok(inverse) = matrix_inverse_with_scratch(&regularized, inverse_scratch)
            .or_else(|_| matrix_inverse(&regularized))
            && matrix_is_finite(&inverse)
        {
            return inverse;
        }
        jitter *= 10.0;
    }

    identity_matrix(dim)
}

pub(super) fn matrix_inverse(matrix: &Mat<f64>) -> Result<Mat<f64>, MtpError> {
    let dim = matrix.ncols();
    let identity = Mat::<f64>::identity(dim, dim);
    solve_linear_system(matrix, &identity).map_err(|_| MtpError::SolveFailed)
}

pub(super) fn matrix_inverse_with_scratch(
    matrix: &Mat<f64>,
    scratch: &mut SpdInverseScratch,
) -> Result<Mat<f64>, MtpError> {
    scratch.invert_spd(matrix)
}

pub(super) fn max_subject_effect_change(current: &[Vec<f64>], previous: &[Vec<f64>]) -> f64 {
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

pub(super) fn max_slice_abs_diff(current: &[f64], previous: &[f64]) -> f64 {
    current
        .iter()
        .zip(previous.iter())
        .map(|(now, before)| (now - before).abs())
        .fold(0.0, f64::max)
}

pub(super) fn linear_predictor(design_matrix: &Mat<f64>, coefficients: &[f64]) -> Vec<f64> {
    (0..design_matrix.nrows())
        .map(|row| dot_row(design_matrix, row, coefficients))
        .collect()
}

pub(super) fn subject_effect_log_prior(effect: &[f64], random_effects_precision: &Mat<f64>) -> f64 {
    -0.5 * quadratic_form_vec(effect, random_effects_precision)
}

pub(super) fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    rng.sample(rand_distr::StandardNormal)
}

pub(super) fn diagonal_from_matrix(matrix: &Mat<f64>) -> Vec<f64> {
    let n = matrix.nrows().min(matrix.ncols());
    (0..n).map(|idx| matrix[(idx, idx)]).collect()
}

pub(super) fn identity_matrix(dim: usize) -> Mat<f64> {
    Mat::identity(dim, dim)
}

pub(super) fn column_to_vec(column: &Mat<f64>) -> Vec<f64> {
    (0..column.nrows()).map(|row| column[(row, 0)]).collect()
}

pub(super) struct SpdInverseScratch {
    dim: usize,
    factor: Mat<f64>,
    rhs: Mat<f64>,
    scratch: MemBuffer,
}

impl SpdInverseScratch {
    #[must_use]
    pub(super) fn new(dim: usize) -> Self {
        let factor_mem =
            llt::factor::cholesky_in_place_scratch::<f64>(dim, Par::Seq, Spec::default());
        let solve_mem = llt::solve::solve_in_place_scratch::<f64>(dim, dim, Par::Seq);
        Self {
            dim,
            factor: Mat::zeros(dim, dim),
            rhs: Mat::zeros(dim, dim),
            scratch: MemBuffer::new(factor_mem.or(solve_mem)),
        }
    }

    pub(super) fn invert_spd(&mut self, matrix: &Mat<f64>) -> Result<Mat<f64>, MtpError> {
        if matrix.nrows() != self.dim || matrix.ncols() != self.dim {
            return Err(MtpError::SolveFailed);
        }

        self.factor.as_mut().copy_from(matrix.as_ref());
        for row in 0..self.dim {
            for col in 0..self.dim {
                self.rhs[(row, col)] = if row == col { 1.0 } else { 0.0 };
            }
        }

        let factor_stack = MemStack::new(&mut self.scratch);
        if llt::factor::cholesky_in_place(
            self.factor.as_mut(),
            LltRegularization::default(),
            Par::Seq,
            factor_stack,
            Spec::default(),
        )
        .is_err()
        {
            return Err(MtpError::SolveFailed);
        }

        let solve_stack = MemStack::new(&mut self.scratch);
        llt::solve::solve_in_place_with_conj(
            self.factor.as_ref(),
            faer::Conj::No,
            self.rhs.as_mut(),
            Par::Seq,
            solve_stack,
        );

        if !matrix_is_finite(&self.rhs) {
            return Err(MtpError::SolveFailed);
        }
        Ok(self.rhs.clone())
    }
}
