use faer::Mat;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use semicontinuous_models::{
    LongitudinalModelInput, MtpFitOptions, MtpProposalTuning, MtpSamplerConfig,
    fit_mtp_input_with_posterior_config, posterior_predictive_summary,
};

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = (1.0_f64 - rng.random::<f64>()).max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0_f64 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn centered_time(period: usize, periods: usize) -> f64 {
    if periods <= 1 {
        0.0
    } else {
        2.0f64.mul_add(usize_to_f64(period) / usize_to_f64(periods - 1), -1.0)
    }
}

fn simulate_mtp_panel(
    n_subjects: usize,
    periods: usize,
    alpha: [f64; 2],
    beta: [f64; 2],
    omega_sq: f64,
    seed: u64,
) -> LongitudinalModelInput {
    let nrows = n_subjects * periods;
    let omega = omega_sq.sqrt();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut outcome = Vec::with_capacity(nrows);
    let mut x_binary = Vec::with_capacity(nrows * 2);
    let mut x_mean = Vec::with_capacity(nrows * 2);
    let mut subject_ids = Vec::with_capacity(nrows);
    let mut time = Vec::with_capacity(nrows);

    for subject in 0..n_subjects {
        for period in 0..periods {
            let t = centered_time(period, periods);
            let eta_binary = alpha[1].mul_add(t, alpha[0]);
            let p_positive = logistic(eta_binary);
            let marginal_log_mean = beta[1].mul_add(t, beta[0]);
            let xi = (-0.5 * omega_sq).mul_add(1.0, marginal_log_mean - p_positive.ln());

            let y = if rng.random::<f64>() < p_positive {
                (omega.mul_add(sample_standard_normal(&mut rng), xi)).exp()
            } else {
                0.0
            };

            outcome.push(y);
            x_binary.extend_from_slice(&[1.0, t]);
            x_mean.extend_from_slice(&[1.0, t]);
            subject_ids.push(u64::try_from(subject + 1).unwrap_or(u64::MAX));
            time.push(usize_to_f64(period));
        }
    }

    LongitudinalModelInput::new(
        Mat::from_fn(nrows, 1, |row, _| outcome[row]),
        Mat::from_fn(nrows, 2, |row, col| x_binary[2 * row + col]),
        Mat::from_fn(nrows, 2, |row, col| x_mean[2 * row + col]),
        subject_ids,
        time,
    )
}

fn zero_rate(input: &LongitudinalModelInput) -> f64 {
    let zeros = (0..input.outcome.nrows())
        .filter(|&row| input.outcome[(row, 0)] <= 0.0)
        .count();
    usize_to_f64(zeros) / usize_to_f64(input.outcome.nrows())
}

fn high_tail_ratio(input: &LongitudinalModelInput, threshold: f64) -> f64 {
    let positives = (0..input.outcome.nrows())
        .filter(|&row| input.outcome[(row, 0)] > threshold)
        .count();
    usize_to_f64(positives) / usize_to_f64(input.outcome.nrows())
}

#[test]
fn mtp_recovers_key_parameters_on_synthetic_data() {
    let true_alpha = [-0.35, 0.8];
    let true_beta = [1.2, 0.55];
    let true_omega_sq = 0.45;

    let input = simulate_mtp_panel(120, 4, true_alpha, true_beta, true_omega_sq, 42);
    let config = MtpSamplerConfig {
        fit_options: MtpFitOptions {
            iterations: 420,
            burn_in: 120,
            thin: 6,
            ..MtpFitOptions::default()
        },
        proposal_tuning: MtpProposalTuning {
            adaptation_interval: 40,
            ..MtpProposalTuning::default()
        },
        ..MtpSamplerConfig::default()
    };

    let (_model, report, _posterior) =
        fit_mtp_input_with_posterior_config(&input, config).expect("fit should run");
    let summary = report
        .posterior_summary
        .expect("posterior summary should be present");

    assert_eq!(summary.alpha.len(), 2);
    assert_eq!(summary.beta.len(), 2);
    assert!((summary.alpha[1].mean - true_alpha[1]).abs() < 0.6);
    assert!((summary.beta[1].mean - true_beta[1]).abs() < 0.45);
    let omega_mean = summary.omega_sq.expect("omega summary").mean;
    assert!(omega_mean.is_finite());
    assert!(omega_mean > 0.05);
    assert!(omega_mean < 3.0);
}

#[test]
fn mtp_stays_stable_under_extreme_zero_inflation() {
    let input = simulate_mtp_panel(140, 3, [-3.5, -0.4], [1.0, 0.2], 0.8, 7);
    let observed_zero_rate = zero_rate(&input);
    assert!(observed_zero_rate > 0.9);

    let config = MtpSamplerConfig {
        fit_options: MtpFitOptions {
            iterations: 320,
            burn_in: 80,
            thin: 4,
            ..MtpFitOptions::default()
        },
        ..MtpSamplerConfig::default()
    };

    let (_model, report, posterior) =
        fit_mtp_input_with_posterior_config(&input, config).expect("fit should run");
    assert!(!posterior.draws.is_empty());
    assert!(report.posterior_summary.is_some());

    let ppc = posterior_predictive_summary(&posterior, &input, 10).expect("ppc should run");
    assert!(ppc.predicted_zero_rate.mean.is_finite());
    assert!(ppc.predicted_zero_rate.mean > 0.75);
    assert!(ppc.predicted_mean.mean.is_finite());
}

#[test]
fn mtp_stays_stable_under_highly_skewed_positive_tail() {
    let input = simulate_mtp_panel(120, 4, [-0.2, 0.25], [0.9, 0.35], 2.4, 99);
    assert!(high_tail_ratio(&input, 30.0) > 0.01);

    let config = MtpSamplerConfig {
        fit_options: MtpFitOptions {
            iterations: 320,
            burn_in: 80,
            thin: 4,
            ..MtpFitOptions::default()
        },
        proposal_tuning: MtpProposalTuning {
            log_omega_draw_scale: 0.02,
            ..MtpProposalTuning::default()
        },
        ..MtpSamplerConfig::default()
    };

    let (_model, report, posterior) =
        fit_mtp_input_with_posterior_config(&input, config).expect("fit should run");
    let summary = report
        .posterior_summary
        .expect("posterior summary should be present");
    assert!(summary.omega_sq.expect("omega summary").mean > 0.5);

    let ppc = posterior_predictive_summary(&posterior, &input, 10).expect("ppc should run");
    assert!(ppc.predicted_mean.mean.is_finite());
    assert!(ppc.brier_score.mean.is_finite());
}
