//! Coordinate-descent weighted least squares for the elastic-net penalty.

use faer::Mat;

use super::{Regularization, TwoPartError};

pub(super) fn elastic_net_wls(
    x: &Mat<f64>,
    weights: &Mat<f64>,
    z: &Mat<f64>,
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
