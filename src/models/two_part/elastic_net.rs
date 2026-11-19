//! Coordinate-descent weighted least squares for the elastic-net penalty.

use faer::{Mat, MatRef};

use super::{Regularization, TwoPartError};

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
    let mut beta = Mat::<f64>::zeros(p, 1);
    let mut residual = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        residual[(i, 0)] = z[(i, 0)];
    }

    let mut col_norms = vec![0.0; p];
    for j in 0..p {
        let mut norm = 0.0;
        for i in 0..n {
            let xij = x[(i, j)];
            norm = (weights[(i, 0)] * xij).mul_add(xij, norm);
        }
        col_norms[j] = norm.max(1e-12);
    }

    let mut iterations = 0;
    loop {
        iterations += 1;
        let mut max_delta = 0.0;
        for j in 0..p {
            let mut rho = 0.0;
            for i in 0..n {
                rho = (weights[(i, 0)] * x[(i, j)]).mul_add(residual[(i, 0)], rho);
            }
            rho = col_norms[j].mul_add(beta[(j, 0)], rho);

            let new_beta = if exclude_intercept && j == 0 {
                rho / col_norms[j]
            } else {
                soft_threshold(rho, lambda * alpha) / lambda.mul_add(1.0 - alpha, col_norms[j])
            };

            let delta = new_beta - beta[(j, 0)];
            if delta != 0.0 {
                for i in 0..n {
                    residual[(i, 0)] = x[(i, j)].mul_add(-delta, residual[(i, 0)]);
                }
            }
            if delta.abs() > max_delta {
                max_delta = delta.abs();
            }
            beta[(j, 0)] = new_beta;
        }

        if max_delta < 1e-6 || iterations >= 200 {
            break;
        }
    }

    Ok(beta)
}

fn soft_threshold(value: f64, penalty: f64) -> f64 {
    if value > penalty {
        value - penalty
    } else if value < -penalty {
        value + penalty
    } else {
        0.0
    }
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

    /// The coordinate descent against `glmnet(family = "gaussian", weights = w,
    /// standardize = FALSE)` on the same design, with lambda and alpha mapped to
    /// glmnet's scaling as described in `tests/r/elastic_net_reference.R`. The descent
    /// stops when no coefficient moves by 1e-6, which accounts for the deviation of a
    /// few times 1e-7 from glmnet's converged solution.
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
        assert!(worst < 2e-6, "worst deviation {worst}");
        assert!(worst_kkt < 1e-4, "worst KKT violation {worst_kkt}");
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
