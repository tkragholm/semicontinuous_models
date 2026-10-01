//! Model-based and sandwich covariance of the two parts.

use faer::Mat;

use super::irls::{logistic, ridge_from_regularization};
use super::{FitOptions, TwoPartError};
use crate::models::covariance::{sandwich, score_meat};
use crate::models::matrix_ops::map_mat;
use crate::utils::{
    add_ridge_to_diagonal, linear_predictor, solve_linear_system, solve_linear_system_ref,
    weighted_xtx,
};

/// Logit-part covariance: model-based, or the sandwich when `robust_se` is set, with
/// cluster-robust meat when `clusters` is given.
pub(super) fn covariance_logit(
    x: &Mat<f64>,
    y: &Mat<f64>,
    weights: &Mat<f64>,
    beta: &Mat<f64>,
    clusters: Option<&[u64]>,
    options: FitOptions,
) -> Result<Mat<f64>, TwoPartError> {
    let eta = linear_predictor(x, beta);
    let p = logistic(&eta);
    let weights = Mat::from_fn(p.nrows(), 1, |i, _| {
        let value = p[(i, 0)] * (1.0 - p[(i, 0)]);
        (value.max(options.min_weight)) * weights[(i, 0)]
    });

    let mut xtwx = weighted_xtx(x, &weights);
    if let Some((lambda, exclude_intercept)) = ridge_from_regularization(options.regularization)
        && lambda > 0.0
    {
        add_ridge_to_diagonal(&mut xtwx, lambda, exclude_intercept);
    }
    if !options.robust_se {
        return covariance_from_information(&xtwx);
    }

    let residuals = Mat::from_fn(y.nrows(), 1, |i, _| {
        (y[(i, 0)] - p[(i, 0)]) * weights[(i, 0)]
    });
    let (meat, _) = score_meat(x, |i| residuals[(i, 0)], clusters);
    sandwich(&xtwx, &meat, solve_linear_system_ref)
}

/// Gamma-part covariance: model-based, or the sandwich when `robust_se` is set, with
/// cluster-robust meat when `clusters` is given.
pub(super) fn covariance_gamma(
    x: &Mat<f64>,
    y: &Mat<f64>,
    weights: &Mat<f64>,
    beta: &Mat<f64>,
    clusters: Option<&[u64]>,
    options: FitOptions,
) -> Result<Mat<f64>, TwoPartError> {
    let eta = linear_predictor(x, beta);
    let mu = map_mat(&eta, f64::exp);

    let mut xtx = weighted_xtx(x, weights);
    if let Some((lambda, exclude_intercept)) = ridge_from_regularization(options.regularization)
        && lambda > 0.0
    {
        add_ridge_to_diagonal(&mut xtx, lambda, exclude_intercept);
    }
    if !options.robust_se {
        return covariance_from_information(&xtx);
    }

    let residuals = Mat::from_fn(y.nrows(), 1, |i, _| {
        ((y[(i, 0)] - mu[(i, 0)]) / mu[(i, 0)]) * weights[(i, 0)]
    });
    let (meat, _) = score_meat(x, |i| residuals[(i, 0)], clusters);
    sandwich(&xtx, &meat, solve_linear_system_ref)
}

fn covariance_from_information(information: &Mat<f64>) -> Result<Mat<f64>, TwoPartError> {
    let identity = Mat::<f64>::identity(information.nrows(), information.ncols());
    solve_linear_system(information, &identity).map_err(|_| TwoPartError::SolveFailed)
}
