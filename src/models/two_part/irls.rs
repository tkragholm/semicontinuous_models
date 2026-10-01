//! The IRLS fits of the logit and gamma log-link parts.

use faer::Mat;

use super::elastic_net::elastic_net_wls;
use super::{FitOptions, Regularization, TwoPartError};
use crate::models::matrix_ops::{
    center_beta, center_columns, map_mat, max_abs_linear_predictor, uncenter_beta,
    weighted_column_means,
};
use crate::utils::{
    CachedFactor, add_ridge_to_diagonal, linear_predictor, matvec_into, max_abs_diff, mean_column,
    weighted_xtx, weighted_xtz_with_buffer,
};

pub(super) fn fit_logit_weighted(
    x: &Mat<f64>,
    y: &Mat<f64>,
    weights: &Mat<f64>,
    options: FitOptions,
    initial_beta: Option<&Mat<f64>>,
) -> Result<(Mat<f64>, usize), TwoPartError> {
    // Degenerate outcome guards:
    // If the binary outcome is constant (all 0 or all 1), IRLS will not
    // meaningfully iterate toward a finite optimum. In this case we fit an
    // intercept-only prevalence model with a clamped logit.
    let mut weighted_sum = 0.0;
    let mut weighted_positive = 0.0;
    let mut saw_positive = false;
    let mut saw_zero = false;
    let mut count_positive = 0usize;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        if yi > 0.0 {
            saw_positive = true;
            count_positive += 1;
        } else {
            saw_zero = true;
        }
        let wi = weights[(i, 0)];
        weighted_sum += wi;
        weighted_positive = yi.mul_add(wi, weighted_positive);
    }
    // Minority-class count, which decides whether a logit is identified at all.
    let minority = count_positive.min(y.nrows() - count_positive);

    // A constant outcome has no logit to fit. NEITHER does one whose minority
    // class is no larger than the parameter count: with p parameters and m <= p
    // observations in the smaller class a separating hyperplane always exists, so
    // the MLE is at infinity and IRLS chases it until `max_iter`.
    //
    // Measured on the SCD cost components: `primary_sector` has 0 zeros in 1664,
    // hits the constant-outcome branch below, and fits in 0.10s with no failures.
    // `lmdb` and `lpr_kontakt` have ONE zero in 1664 against 31 parameters, miss
    // the branch, and spend 4.5s each while 45% of their bootstrap replicates
    // report non-convergence. `lpr_sghforlob` has two, and every one of its 1000
    // replicates "converged" to the identical coefficient 349.9 -- a fitted
    // probability of 1 to machine precision, and a logit contributing no
    // bootstrap variability whatever.
    //
    // All four are the same degeneracy. This widens the guard from "no variation"
    // to "not enough variation to identify a slope", and they all take the
    // intercept-only prevalence model that the first case already took.
    if !saw_positive || !saw_zero || minority <= x.ncols() {
        let mut beta = Mat::<f64>::zeros(x.ncols(), 1);
        if weighted_sum > 0.0 && x.ncols() > 0 {
            let p = (weighted_positive / weighted_sum)
                .clamp(options.min_weight, 1.0 - options.min_weight);
            beta[(0, 0)] = (p / (1.0 - p)).ln();
        }
        return Ok((beta, 0));
    }

    // Warm-start from the full-sample logit coefficients when supplied; else start at 0.
    let mut beta = match initial_beta {
        Some(init) if init.nrows() == x.ncols() && init.ncols() == 1 => init.clone(),
        _ => Mat::<f64>::zeros(x.ncols(), 1),
    };
    let regularization = options.regularization;
    let mut weighted_xtz_buffer = Vec::new();

    let prior = weights;
    let mut eta = logit_eta(x, &beta);
    let mut p = logistic(&eta);
    let mut deviance = logit_deviance(y, &p, prior);

    for iteration in 0..options.max_iter {
        // IRLS weight: the prior weight times the variance function, floored so a
        // fitted probability at 0 or 1 cannot make `X'WX` singular.
        let irls_weights = Mat::from_fn(p.nrows(), 1, |i, _| {
            let variance = p[(i, 0)] * (1.0 - p[(i, 0)]);
            variance.max(options.min_weight) * prior[(i, 0)]
        });

        // Working response. The residual is divided by the VARIANCE FUNCTION, not
        // by the IRLS weight: dividing by `variance * prior` cancels the prior
        // weight out of `X'Wz`, leaving the score equation `X'(y - p) = 0` -- the
        // UNWEIGHTED one. The gamma loop below already divides by its variance
        // term rather than by `W`; this makes the two agree. Inert while every
        // prior weight is 1, which is the case whenever the panel carries no
        // weight column, and wrong the moment one does.
        let z = Mat::from_fn(eta.nrows(), 1, |i, _| {
            let variance = (p[(i, 0)] * (1.0 - p[(i, 0)])).max(options.min_weight);
            eta[(i, 0)] + (y[(i, 0)] - p[(i, 0)]) / variance
        });

        let proposal = weighted_least_squares(
            x,
            &irls_weights,
            &z,
            regularization,
            &mut weighted_xtz_buffer,
        )?;
        let step = &proposal - &beta;

        // Step-halving, as `glm.fit` does: a full Newton step that makes the
        // deviance worse (or non-finite) is halved until it does not. Without it
        // the step is taken unconditionally and a near-separated logit oscillates
        // until `max_iter`.
        let mut halvings = 0usize;
        let (beta_next, eta_next, p_next, deviance_next) = loop {
            let scale = 0.5f64.powi(i32::try_from(halvings).unwrap_or(i32::MAX));
            let candidate = &beta + &(step.as_ref() * scale);
            let candidate_eta = logit_eta(x, &candidate);
            let candidate_p = logistic(&candidate_eta);
            let candidate_deviance = logit_deviance(y, &candidate_p, prior);
            if candidate_deviance.is_finite() && candidate_deviance <= deviance {
                break (candidate, candidate_eta, candidate_p, candidate_deviance);
            }
            halvings += 1;
            if halvings > options.max_step_halvings {
                // No downhill step exists at this point. That is convergence to a
                // stationary point, not a failure: report the current iterate.
                return Ok((beta, iteration + 1));
            }
        };

        // Relative change in deviance, `glm.fit`'s criterion. The absolute
        // coefficient test is kept as a secondary: it is what a well-conditioned
        // fit used to stop on, and keeping it means such a fit stops no later
        // than it did before.
        let converged_deviance = (deviance_next - deviance).abs() / (deviance_next.abs() + 0.1)
            < options.deviance_tolerance;
        let converged_step = max_abs_diff(&beta_next, &beta) < options.tolerance;

        beta = beta_next;
        eta = eta_next;
        p = p_next;
        deviance = deviance_next;

        if converged_deviance || converged_step {
            return Ok((beta, iteration + 1));
        }
    }

    Err(TwoPartError::NonConvergence)
}

/// Bound on the logit linear predictor.
///
/// `logistic(36)` is already 1.0 in f64, so beyond this the fitted probability
/// carries no information and only the working response keeps growing. The gamma
/// loop has had such a clamp since heavy-tailed outcomes were found to drive
/// `exp(eta)` to infinity; the logit loop -- the one that actually diverges on
/// this data -- had none.
const LOGIT_ETA_CLAMP: f64 = 40.0;

fn logit_eta(x: &Mat<f64>, beta: &Mat<f64>) -> Mat<f64> {
    let eta = linear_predictor(x, beta);
    map_mat(&eta, |value| value.clamp(-LOGIT_ETA_CLAMP, LOGIT_ETA_CLAMP))
}

/// Weighted binomial deviance, `-2 sum w_i [y ln p + (1-y) ln(1-p)]`.
///
/// `p` is clamped off 0 and 1 so a saturated fit gives a large finite deviance
/// rather than an infinity that no comparison can order.
fn logit_deviance(y: &Mat<f64>, p: &Mat<f64>, weights: &Mat<f64>) -> f64 {
    const EPS: f64 = 1e-12;
    let mut deviance = 0.0;
    for i in 0..y.nrows() {
        let prob = p[(i, 0)].clamp(EPS, 1.0 - EPS);
        let outcome = f64::from(y[(i, 0)] > 0.0);
        let term = outcome.mul_add(prob.ln(), (1.0 - outcome) * (1.0 - prob).ln());
        deviance = weights[(i, 0)].mul_add(-2.0 * term, deviance);
    }
    deviance
}

/// Bound on the gamma log-link linear predictor during IRLS and prediction.
/// `mu = exp(eta)`; clamping `eta` to ±30 keeps `mu` in roughly [9e-14, 1e13],
/// orders of magnitude beyond any plausible fitted mean, so the bound never
/// binds for a well-behaved fit. It only keeps a transient overshoot finite:
/// heavy-tailed positive outcomes can otherwise drive `eta` large enough that
/// `mu` overflows to +inf, making the IRLS working response non-finite and the
/// solve diverge — a failure that most often surfaces as non-convergence inside
/// a resampling bootstrap.
pub(super) const GAMMA_LOG_LINK_ETA_CLAMP: f64 = 30.0;

pub(super) fn fit_gamma_log_link_weighted(
    x: &Mat<f64>,
    y: &Mat<f64>,
    weights: &Mat<f64>,
    options: FitOptions,
    initial_beta: Option<&Mat<f64>>,
) -> Result<(Mat<f64>, usize), TwoPartError> {
    let regularization = options.regularization;
    let mut weighted_xtz_buffer = Vec::new();

    // Center the covariate columns for numerical conditioning (see the module note in
    // `matrix_ops`). IRLS runs on the centered design `cx`, keeping the intercept near
    // `ln(mean y)` and η in the data-supported range instead of drifting into the log-link
    // clamp; the converged coefficients are un-centered back to the raw-x scale on return,
    // so the stored `beta_gamma` and every prediction operate on the original design.
    let col_means = weighted_column_means(x, Some(weights));
    let centered_storage;
    let cx: &Mat<f64> = match &col_means {
        Some(means) => {
            centered_storage = center_columns(x, means);
            &centered_storage
        }
        None => x,
    };

    // Warm-start from the full-sample coefficients when supplied (a bootstrap replicate
    // converges to the same optimum in fewer iterations); otherwise start from the
    // intercept-only model (log of the outcome mean). A supplied warm-start is on the
    // raw-x scale, so map it onto the centered scale first.
    let mut beta = match initial_beta {
        Some(init) if init.nrows() == cx.ncols() && init.ncols() == 1 => match &col_means {
            Some(means) => center_beta(init, means),
            None => init.clone(),
        },
        _ => {
            let mut beta = Mat::<f64>::zeros(cx.ncols(), 1);
            if cx.ncols() > 0 {
                let mean = mean_column(y);
                if mean > 0.0 {
                    beta[(0, 0)] = mean.ln();
                }
            }
            beta
        }
    };

    // The gamma log-link Fisher-scoring weight is the fixed prior weight (`μ^0 == 1`),
    // so `X'WX` is invariant across iterations: only the working-response RHS changes.
    // For ridge / no penalty, factor it once and back-substitute each pass, dropping the
    // O(n·p²) Gram build + O(p³) factorization from every iteration but the first.
    // ElasticNet uses coordinate descent (no single factorization) and keeps the
    // per-iteration path. Exact — the cached factor is the true information matrix.
    let cached = if matches!(regularization, Regularization::ElasticNet { .. }) {
        None
    } else {
        let mut xtwx = weighted_xtx(cx, weights);
        if let Some((lambda, exclude_intercept)) = ridge_from_regularization(regularization)
            && lambda > 0.0
        {
            add_ridge_to_diagonal(&mut xtwx, lambda, exclude_intercept);
        }
        Some(CachedFactor::factor(&xtwx))
    };

    // Column buffers reused across iterations (the bootstrap runs thousands of these
    // fits in parallel; per-iteration allocation churns the allocator).
    let n = cx.nrows();
    let mut eta = Mat::<f64>::zeros(n, 1);
    let mut mu = Mat::<f64>::zeros(n, 1);
    let mut z = Mat::<f64>::zeros(n, 1);

    for iteration in 0..options.max_iter {
        // Clamp the linear predictor before exponentiating so `mu` stays finite
        // and strictly positive on pathological resamples (see the constant). With the
        // centered design this ceiling is a dormant safety net for well-posed data.
        matvec_into(&mut eta, cx, &beta);
        for i in 0..n {
            eta[(i, 0)] = eta[(i, 0)].clamp(-GAMMA_LOG_LINK_ETA_CLAMP, GAMMA_LOG_LINK_ETA_CLAMP);
            mu[(i, 0)] = eta[(i, 0)].exp();
            z[(i, 0)] = eta[(i, 0)] + (y[(i, 0)] - mu[(i, 0)]) / mu[(i, 0)];
        }

        let beta_next = if let Some(factor) = &cached {
            let rhs = weighted_xtz_with_buffer(cx, weights, &z, &mut weighted_xtz_buffer);
            match factor.solve(rhs.as_ref()) {
                Ok(candidate) => candidate,
                // Pathological RHS: fall back to the stabilized per-iteration solve.
                Err(_) => weighted_least_squares(
                    cx,
                    weights,
                    &z,
                    regularization,
                    &mut weighted_xtz_buffer,
                )?,
            }
        } else {
            weighted_least_squares(cx, weights, &z, regularization, &mut weighted_xtz_buffer)?
        };

        // A non-finite step means this resample is pathological at the current
        // regularization level; report non-convergence so the Relaxed strategy
        // retries with stronger ridge rather than propagating NaN downstream.
        if !crate::utils::matrix_is_finite(&beta_next) {
            return Err(TwoPartError::NonConvergence);
        }
        if max_abs_diff(&beta_next, &beta) < options.tolerance {
            // Saturation guard: if the converged fit still reaches the log-link ceiling
            // its predicted means are distorted — report non-convergence so the Relaxed
            // strategy retries with stronger ridge rather than returning a degenerate fit.
            // (η is evaluated on the centered design, but `cx·β̃ == x·β` row-for-row.)
            if max_abs_linear_predictor(cx, &beta_next) >= GAMMA_LOG_LINK_ETA_CLAMP {
                return Err(TwoPartError::NonConvergence);
            }
            // Un-center back to the raw-x scale so the stored coefficients and every
            // prediction operate on the original design.
            let beta_raw = match &col_means {
                Some(means) => uncenter_beta(&beta_next, means),
                None => beta_next,
            };
            return Ok((beta_raw, iteration + 1));
        }
        beta = beta_next;
    }

    Err(TwoPartError::NonConvergence)
}

fn weighted_least_squares(
    x: &Mat<f64>,
    weights: &Mat<f64>,
    z: &Mat<f64>,
    regularization: Regularization,
    weighted_buffer: &mut Vec<f64>,
) -> Result<Mat<f64>, TwoPartError> {
    if let Regularization::ElasticNet { .. } = regularization {
        return elastic_net_wls(x, weights, z, regularization);
    }

    let mut xtwx = weighted_xtx(x, weights);
    if let Some((lambda, exclude_intercept)) = ridge_from_regularization(regularization)
        && lambda > 0.0
    {
        add_ridge_to_diagonal(&mut xtwx, lambda, exclude_intercept);
    }
    let xtw_rhs = weighted_xtz_with_buffer(x, weights, z, weighted_buffer);
    crate::utils::solve_linear_system(&xtwx, &xtw_rhs).map_err(|_| TwoPartError::SolveFailed)
}

pub(super) fn logistic(values: &Mat<f64>) -> Mat<f64> {
    map_mat(values, |value| 1.0 / (1.0 + (-value).exp()))
}

pub(super) fn ridge_from_regularization(regularization: Regularization) -> Option<(f64, bool)> {
    match regularization {
        Regularization::None => None,
        Regularization::Ridge {
            lambda,
            exclude_intercept,
        } => Some((lambda.max(0.0), exclude_intercept)),
        Regularization::ElasticNet {
            lambda,
            alpha,
            exclude_intercept,
        } => {
            let l2 = lambda.max(0.0) * (1.0 - alpha.clamp(0.0, 1.0));
            Some((l2, exclude_intercept))
        }
        Regularization::BayesianRidge {
            prior_scale,
            exclude_intercept,
        } => {
            let scale = prior_scale.max(1e-8);
            Some((1.0 / (scale * scale), exclude_intercept))
        }
    }
}
