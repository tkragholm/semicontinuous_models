use faer::Mat;

#[must_use]
pub fn select_rows(matrix: &Mat<f64>, indices: &[usize]) -> Mat<f64> {
    Mat::from_fn(indices.len(), matrix.ncols(), |i, j| {
        matrix[(indices[i], j)]
    })
}

#[must_use]
pub fn select_values(vector: &Mat<f64>, indices: &[usize]) -> Mat<f64> {
    Mat::from_fn(indices.len(), 1, |i, _| vector[(indices[i], 0)])
}

#[must_use]
pub fn map_mat(values: &Mat<f64>, f: impl Fn(f64) -> f64) -> Mat<f64> {
    Mat::from_fn(values.nrows(), values.ncols(), |i, j| f(values[(i, j)]))
}

// ── Design-matrix centering ──────────────────────────────────────────────────────────
//
// Log-link GLMs (Tweedie / gamma) are fit on an UNCENTERED design whose intercept column
// then absorbs the whole offset — pushing the fitted intercept far above `ln(mean y)`
// (observed: 27–78 on cost data). When the intercept nears the `exp_clamped` saturation
// ceiling (η = ±30), IRLS iterates and the marginal-standardisation prediction both start
// clamping η, which corrupts the fit (a degenerate "converged" solution) and collapses the
// counterfactual means (exposed ≈ unexposed, ratio → 1 instead of exp(β_exposed)).
//
// Centering the non-intercept columns is a reparametrisation that leaves `Xβ`, the fitted
// means, and the deviance invariant, but keeps the intercept at ≈ `ln(mean y)` and the
// information matrix well-conditioned, so η stays in the data-supported range and the clamp
// never fires. Callers fit on the centered design and un-center the coefficients back to the
// raw-x scale (:func:`uncenter_beta`) before computing SE/cov, predictions, and marginal
// effects — so nothing downstream sees the centered scale.

const CONSTANT_COLUMN_TOL: f64 = 1e-12;

fn column_is_constant(x: &Mat<f64>, column: usize) -> bool {
    let n = x.nrows();
    if n == 0 {
        return true;
    }
    let first = x[(0, column)];
    (1..n).all(|i| (x[(i, column)] - first).abs() <= CONSTANT_COLUMN_TOL)
}

/// Sample-weighted column means for centering the design, or `None` when centering must be
/// skipped (empty design, or column 0 is not a constant intercept — with no free intercept
/// to absorb the shift, centering would change the fitted model).
///
/// `means[0]` is `0.0` (the intercept is never centered); any other CONSTANT column also
/// gets `0.0` so centering leaves it intact (centering a constant column would zero it and
/// make `XᵀWX` singular). Every remaining column gets its weighted mean.
#[must_use]
pub(crate) fn weighted_column_means(
    x: &Mat<f64>,
    weights: Option<&Mat<f64>>,
) -> Option<Vec<f64>> {
    let (n, p) = (x.nrows(), x.ncols());
    if n == 0 || p == 0 || !column_is_constant(x, 0) {
        return None;
    }
    let weight_sum: f64 = weights.map_or(n as f64, |w| (0..n).map(|i| w[(i, 0)]).sum());
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return None;
    }
    let mut means = vec![0.0_f64; p];
    for j in 1..p {
        if column_is_constant(x, j) {
            continue;
        }
        let mut acc = 0.0_f64;
        for i in 0..n {
            let w = weights.map_or(1.0, |weights| weights[(i, 0)]);
            acc = w.mul_add(x[(i, j)], acc);
        }
        means[j] = acc / weight_sum;
    }
    Some(means)
}

/// Return `x` with each column shifted by `-means[j]` (the intercept and any constant
/// column carry `means[j] == 0.0`, so they pass through unchanged).
#[must_use]
pub(crate) fn center_columns(x: &Mat<f64>, means: &[f64]) -> Mat<f64> {
    Mat::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] - means[j])
}

/// Map centered-scale coefficients back to the raw-x scale. Slopes are unchanged; the
/// intercept (column 0) absorbs `-Σ_{j≥1} β̃ⱼ·means[j]` so that `Xβ == X̃β̃` for every row.
#[must_use]
pub(crate) fn uncenter_beta(beta_centered: &Mat<f64>, means: &[f64]) -> Mat<f64> {
    let mut beta = beta_centered.clone();
    let shift: f64 = (1..beta.nrows()).map(|j| beta_centered[(j, 0)] * means[j]).sum();
    beta[(0, 0)] -= shift;
    beta
}

/// Inverse of :func:`uncenter_beta`: map raw-scale coefficients (e.g. a warm-start) onto the
/// centered scale so IRLS can resume from them. Slopes are unchanged; column 0 gains
/// `+Σ_{j≥1} βⱼ·means[j]`.
#[must_use]
pub(crate) fn center_beta(beta_raw: &Mat<f64>, means: &[f64]) -> Mat<f64> {
    let mut beta = beta_raw.clone();
    let shift: f64 = (1..beta.nrows()).map(|j| beta_raw[(j, 0)] * means[j]).sum();
    beta[(0, 0)] += shift;
    beta
}

/// Largest `|η|` over the design rows for a fitted `beta` — used to detect log-link
/// saturation: a converged fit whose linear predictor reaches the `exp_clamped` ceiling
/// produces distorted, non-multiplicative predictions, so callers treat it as
/// non-convergence rather than silently reporting a degenerate "ok" fit.
#[must_use]
pub(crate) fn max_abs_linear_predictor(x: &Mat<f64>, beta: &Mat<f64>) -> f64 {
    let eta = x * beta;
    (0..eta.nrows()).fold(0.0_f64, |acc, i| acc.max(eta[(i, 0)].abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Design with an intercept column 0, a binary exposure column 1, and a large-offset
    // continuous covariate column 2 — the shape that makes the uncentered fit ill-conditioned.
    fn sample_design() -> Mat<f64> {
        Mat::from_fn(12, 3, |i, j| match j {
            0 => 1.0,
            1 => f64::from(u32::try_from(i % 2).unwrap()),
            _ => 1000.0 + f64::from(u32::try_from(i).unwrap()),
        })
    }

    #[test]
    fn centering_leaves_the_linear_predictor_invariant() {
        // The core correctness guarantee: fitting on the centered design and un-centering
        // the coefficients back reproduces X·β EXACTLY row-for-row. `beta_centered` stands
        // in for whatever the IRLS converges to on the centered scale.
        let x = sample_design();
        let means = weighted_column_means(&x, None).expect("intercept present");
        let cx = center_columns(&x, &means);
        let beta_centered = Mat::from_fn(3, 1, |i, _| [9.5, 0.7, -0.03][i]);
        let beta_raw = uncenter_beta(&beta_centered, &means);

        let eta_centered = &cx * &beta_centered;
        let eta_raw = &x * &beta_raw;
        for i in 0..x.nrows() {
            assert!(
                (eta_centered[(i, 0)] - eta_raw[(i, 0)]).abs() < 1e-9,
                "row {i}: centered η {} != raw η {}",
                eta_centered[(i, 0)],
                eta_raw[(i, 0)]
            );
        }
    }

    #[test]
    fn center_beta_is_the_inverse_of_uncenter_beta() {
        let x = sample_design();
        let means = weighted_column_means(&x, None).expect("intercept present");
        let beta_raw = Mat::from_fn(3, 1, |i, _| [30.5, 0.7, -0.03][i]);
        let round_trip = uncenter_beta(&center_beta(&beta_raw, &means), &means);
        for i in 0..beta_raw.nrows() {
            assert!((round_trip[(i, 0)] - beta_raw[(i, 0)]).abs() < 1e-9);
        }
    }

    #[test]
    fn weighted_column_means_exempts_intercept_and_constant_columns() {
        // col 0 = intercept (exempt), col 1 = constant 5.0 (exempt), col 2 = varying.
        let x = Mat::from_fn(4, 3, |i, j| match j {
            0 => 1.0,
            1 => 5.0,
            _ => f64::from(u32::try_from(i).unwrap()), // 0,1,2,3 -> mean 1.5
        });
        let means = weighted_column_means(&x, None).expect("intercept present");
        assert!((means[0] - 0.0).abs() < 1e-12, "intercept must not be centered");
        assert!((means[1] - 0.0).abs() < 1e-12, "constant column must not be centered");
        assert!((means[2] - 1.5).abs() < 1e-12, "varying column mean");
    }

    #[test]
    fn weighted_column_means_skips_when_no_intercept() {
        // Column 0 is not constant -> no usable intercept -> centering must be skipped.
        let x = Mat::from_fn(4, 2, |i, j| f64::from(u32::try_from(i + j).unwrap()));
        assert!(weighted_column_means(&x, None).is_none());
    }

    #[test]
    fn weighted_column_means_uses_weights() {
        let x = Mat::from_fn(3, 2, |i, j| if j == 0 { 1.0 } else { [0.0, 10.0, 20.0][i] });
        let w = Mat::from_fn(3, 1, |i, _| [1.0, 1.0, 2.0][i]); // weighted mean = (0+10+40)/4 = 12.5
        let means = weighted_column_means(&x, Some(&w)).expect("intercept present");
        assert!((means[1] - 12.5).abs() < 1e-12);
    }
}
