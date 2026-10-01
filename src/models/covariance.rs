//! Sandwich covariance shared by the two-part, Tweedie and log-normal models.

use std::collections::HashMap;

use faer::Mat;

use crate::utils::row_scaled_gram;

/// The sandwich meat built from per-row score residuals `r_i = residual(i)`.
///
/// Without clusters this is `Σ_i r_i² x_i x_iᵀ`. With cluster labels it is
/// `Σ_g s_g s_gᵀ` with `s_g = Σ_{i∈g} r_i x_i`, and the number of clusters is
/// returned alongside. Score sums are accumulated in row order and clusters are
/// kept in order of first appearance, so the meat does not depend on hashing.
pub(crate) fn score_meat(
    x: &Mat<f64>,
    residual: impl Fn(usize) -> f64 + Sync,
    clusters: Option<&[u64]>,
) -> (Mat<f64>, Option<usize>) {
    let Some(clusters) = clusters else {
        return (row_scaled_gram(x, residual), None);
    };
    let p = x.ncols();
    let mut index_of: HashMap<u64, usize> = HashMap::new();
    let mut sums: Vec<f64> = Vec::new();
    for (row, &label) in clusters.iter().enumerate().take(x.nrows()) {
        let next = index_of.len();
        let group = *index_of.entry(label).or_insert(next);
        if group == next {
            sums.resize(sums.len() + p, 0.0);
        }
        let r = residual(row);
        let sum = &mut sums[group * p..(group + 1) * p];
        for (col, value) in sum.iter_mut().enumerate() {
            *value = r.mul_add(x[(row, col)], *value);
        }
    }
    let n_clusters = index_of.len();
    let sums = Mat::from_fn(n_clusters, p, |group, col| sums[group * p + col]);
    (row_scaled_gram(&sums, |_| 1.0), Some(n_clusters))
}
