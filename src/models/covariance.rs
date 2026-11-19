//! Sandwich covariance shared by the two-part, Tweedie and log-normal models.

use epistat_inference::Clusters;
use epistat_inference::sandwich as epistat_sandwich;
use epistat_linalg::Attempt;
use faer::{Mat, MatRef};

/// The sandwich meat built from per-row score residuals `r_i = residual(i)`.
///
/// Without clusters this is `Σ_i r_i² x_i x_iᵀ`. With cluster labels it is
/// `Σ_g s_g s_gᵀ` with `s_g = Σ_{i∈g} r_i x_i`, and the number of clusters is
/// returned alongside. Clusters are numbered in order of first appearance and score
/// sums are accumulated in row order, so the meat does not depend on hashing.
///
/// # Panics
///
/// Panics if `clusters` does not have one label per row of `x`. The fitters validate
/// this before they get here.
pub(crate) fn score_meat(
    x: MatRef<'_, f64>,
    residual: impl Fn(usize) -> f64 + Sync,
    clusters: Option<&[u64]>,
) -> (Mat<f64>, Option<usize>) {
    let clusters = clusters.map(Clusters::from_labels);
    let meat =
        epistat_sandwich::meat(x, residual, clusters.as_ref()).expect("one cluster label per row");
    (meat, clusters.map(|clusters| clusters.count()))
}

/// `A⁻¹ M A⁻ᵀ` for the information matrix `A` and meat `M`, through two solves with
/// the symmetric fallback sequence. `None` when no factorisation gives a finite solution.
pub(crate) fn sandwich(information: MatRef<'_, f64>, meat: MatRef<'_, f64>) -> Option<Mat<f64>> {
    epistat_sandwich::sandwich(information, meat, &Attempt::SYMMETRIC).ok()
}

/// Standard errors from a covariance matrix, with negative variances read as zero.
pub(crate) fn diag_sqrt(cov: &Mat<f64>) -> Mat<f64> {
    let errors = epistat_sandwich::standard_errors(cov.as_ref());
    Mat::from_fn(errors.len(), 1, |i, _| errors[i])
}
