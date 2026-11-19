//! Weighted least squares with the elastic-net penalty, fitted by
//! [`epistat_glm::ElasticNet`] with the Gaussian family.

use epistat_glm::{ElasticNet, Family, ResponseScaling};
use faer::{Mat, MatRef};

use super::{Regularization, TwoPartError};

/// The bound on the largest variance-weighted squared coefficient change.
const TOLERANCE: f64 = 1e-14;

/// Minimises `0.5 * sum w_i (z_i - x_i b)^2 + lambda * (alpha * |b|_1 + (1 - alpha) / 2 *
/// |b|_2^2)` over the columns of `x` as given, with the first column unpenalised when
/// `exclude_intercept` is set.
///
/// The penalty is on the sum scale of the weights. The fit is `ElasticNet`'s, whose
/// penalty is per unit of weight, so it is run with `lambda / sum(w)` and
/// [`ResponseScaling::None`], which leaves the response unscaled. glmnet's own Gaussian
/// objective divides the ridge term by the weighted standard deviation of `z`.
/// A fit that reaches the pass limit without meeting the tolerance is returned as it
/// stands.
pub(super) fn elastic_net_wls(
    x: MatRef<'_, f64>,
    weights: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    regularization: Regularization,
) -> Result<Mat<f64>, TwoPartError> {
    let (lambda, alpha, exclude_intercept) = match regularization {
        Regularization::ElasticNet {
            lambda,
            alpha,
            exclude_intercept,
        } => (lambda.max(0.0), alpha.clamp(0.0, 1.0), exclude_intercept),
        _ => return Err(TwoPartError::SolveFailed),
    };

    let n = x.nrows();
    let p = x.ncols();
    let w: Vec<f64> = (0..n).map(|i| weights[(i, 0)]).collect();
    let target: Vec<f64> = (0..n).map(|i| z[(i, 0)]).collect();
    let total_weight: f64 = w.iter().sum();
    if !(total_weight.is_finite() && total_weight > 0.0) {
        return Err(TwoPartError::SolveFailed);
    }
    let penalty_factor: Vec<f64> = (0..p)
        .map(|j| {
            if exclude_intercept && j == 0 {
                0.0
            } else {
                1.0
            }
        })
        .collect();

    let mut model = ElasticNet::new(Family::Gaussian, lambda / total_weight, alpha);
    model.tolerance = TOLERANCE;
    model.scaling = ResponseScaling::None;
    let fit = model
        .fit(x, &target, Some(&w), None, Some(&penalty_factor), None)
        .map_err(|_| TwoPartError::SolveFailed)?;
    Ok(Mat::from_fn(p, 1, |j, _| fit.coefficients[j]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Case {
        alpha: f64,
        lambda: f64,
        exclude_intercept: bool,
        beta: Vec<f64>,
    }

    #[derive(Deserialize)]
    struct Fixture {
        x: Vec<Vec<f64>>,
        z: Vec<f64>,
        weights: Vec<f64>,
        cases: Vec<Case>,
    }

    /// The fit against `glmnet(family = "gaussian", weights = w, standardize = FALSE)` on
    /// the same design, with lambda and alpha mapped to glmnet's scaling as described in
    /// `tests/r/elastic_net_reference.R`.
    #[test]
    fn matches_glmnet() {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/reference/elastic_net_glmnet.json"
        ))
        .expect("fixture");
        let fixture: Fixture = serde_json::from_str(&text).expect("fixture parses");
        let n = fixture.z.len();
        let p = fixture.x[0].len();
        let x = Mat::from_fn(n, p, |i, j| fixture.x[i][j]);
        let z = Mat::from_fn(n, 1, |i, _| fixture.z[i]);
        let weights = Mat::from_fn(n, 1, |i, _| fixture.weights[i]);

        let mut worst = 0.0_f64;
        let mut worst_kkt = 0.0_f64;
        for case in &fixture.cases {
            let beta = elastic_net_wls(
                x.as_ref(),
                weights.as_ref(),
                z.as_ref(),
                Regularization::ElasticNet {
                    lambda: case.lambda,
                    alpha: case.alpha,
                    exclude_intercept: case.exclude_intercept,
                },
            )
            .expect("elastic net fit");
            let reference = &case.beta;
            if reference.is_empty() {
                // glmnet cannot fit a penalised constant column. The penalised-intercept
                // fit is checked against the optimality conditions instead.
                worst_kkt = worst_kkt.max(kkt_violation(&x, &z, &weights, &beta, case));
                continue;
            }
            let deviation = (0..p)
                .map(|j| (beta[(j, 0)] - reference[j]).abs())
                .fold(0.0, f64::max);
            worst = worst.max(deviation);
            worst_kkt = worst_kkt.max(kkt_violation(&x, &z, &weights, &beta, case));
        }
        assert!(worst < 1e-9, "worst deviation {worst}");
        assert!(worst_kkt < 1e-5, "worst KKT violation {worst_kkt}");
    }

    /// Largest violation of the subgradient conditions of the penalised objective: for
    /// each penalised coefficient the gradient `-sum w x (z - x b) + lambda (1 - alpha)
    /// b` must equal `-lambda alpha sign(b)` at a non-zero b and lie within
    /// `[-lambda alpha, lambda alpha]` at zero. An unpenalised coefficient has zero
    /// gradient.
    fn kkt_violation(
        x: &Mat<f64>,
        z: &Mat<f64>,
        weights: &Mat<f64>,
        beta: &Mat<f64>,
        case: &Case,
    ) -> f64 {
        let (n, p) = (x.nrows(), x.ncols());
        let mut worst = 0.0_f64;
        for j in 0..p {
            let mut gradient = 0.0;
            for i in 0..n {
                let fitted: f64 = (0..p).map(|k| x[(i, k)] * beta[(k, 0)]).sum();
                gradient -= weights[(i, 0)] * x[(i, j)] * (z[(i, 0)] - fitted);
            }
            let b = beta[(j, 0)];
            let violation = if case.exclude_intercept && j == 0 {
                gradient.abs()
            } else {
                let g = gradient + case.lambda * (1.0 - case.alpha) * b;
                let l1 = case.lambda * case.alpha;
                if b == 0.0 {
                    (g.abs() - l1).max(0.0)
                } else {
                    (g + l1 * b.signum()).abs()
                }
            };
            worst = worst.max(violation);
        }
        worst
    }
}
