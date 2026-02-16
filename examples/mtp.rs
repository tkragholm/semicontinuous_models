use faer::Mat;
use semicontinuous_models::{
    CounterfactualScenario, LongitudinalModelInput, MtpFitOptions, MtpSamplerConfig,
    RandomEffectsStructure, compute_counterfactual_effects_summary,
    fit_mtp_input_with_posterior_config, posterior_predictive_summary,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = build_synthetic_input(24, 4);

    let config = MtpSamplerConfig {
        fit_options: MtpFitOptions {
            iterations: 200,
            burn_in: 80,
            thin: 4,
            seed: 2_026,
            random_effects: RandomEffectsStructure::InterceptsOnly,
            ..MtpFitOptions::default()
        },
        ..MtpSamplerConfig::default()
    };

    let (model, report, posterior) = fit_mtp_input_with_posterior_config(&input, config)?;

    println!(
        "MTP fit complete: binary covariates={}, mean covariates={}, random effect dim={}",
        model.n_binary_covariates, model.n_mean_covariates, model.random_effect_dimension
    );
    println!(
        "Sampler: iterations={}, retained draws={}",
        report.diagnostics.iterations_completed, report.diagnostics.retained_draws
    );

    if let Some(rates) = report.diagnostics.acceptance_rates {
        println!(
            "Acceptance rates: alpha={:.2}, beta={:.2}, random effects={:.2}, kappa={:.2}, omega_sq={:.2}",
            rates.alpha, rates.beta, rates.random_effects, rates.kappa, rates.omega_sq
        );
    }

    if let Some(summary) = &report.posterior_summary {
        println!("Posterior draw count: {}", summary.draw_count);
        if let Some(alpha_intercept) = summary.alpha.first() {
            println!(
                "alpha[intercept] mean={:.3}, 95% CrI [{:.3}, {:.3}]",
                alpha_intercept.mean, alpha_intercept.q025, alpha_intercept.q975
            );
        }
        if let Some(beta_intercept) = summary.beta.first() {
            println!(
                "beta[intercept] mean={:.3}, 95% CrI [{:.3}, {:.3}]",
                beta_intercept.mean, beta_intercept.q025, beta_intercept.q975
            );
        }
    }

    let ppc = posterior_predictive_summary(&posterior, &input, 5)?;
    println!(
        "Posterior predictive: observed zero-rate={:.3}, predicted zero-rate mean={:.3}, brier mean={:.3}",
        ppc.observed_zero_rate, ppc.predicted_zero_rate.mean, ppc.brier_score.mean
    );

    let scenario = build_counterfactual_scenario(4);
    let effects = compute_counterfactual_effects_summary(&posterior, &scenario)?;
    println!(
        "Counterfactual cumulative additive effect mean={:.3}, 95% CrI [{:.3}, {:.3}]",
        effects.cumulative_additive_effect.mean,
        effects.cumulative_additive_effect.q025,
        effects.cumulative_additive_effect.q975
    );

    if let Some(first_period) = effects.per_period.first() {
        println!(
            "Period {} multiplicative effect mean={:.3}, odds-ratio-positive mean={:.3}",
            first_period.period_index,
            first_period.multiplicative_effect.mean,
            first_period.odds_ratio_positive.mean
        );
    }

    Ok(())
}

fn build_synthetic_input(subjects: usize, periods_per_subject: usize) -> LongitudinalModelInput {
    let rows = subjects.saturating_mul(periods_per_subject);
    let subject_ids = (0..rows)
        .map(|row| u64::try_from(row / periods_per_subject).unwrap_or(u64::MAX) + 1)
        .collect::<Vec<_>>();
    let time = (0..rows)
        .map(|row| usize_to_f64(row % periods_per_subject))
        .collect::<Vec<_>>();

    let x_binary = design_matrix(rows, periods_per_subject);
    let x_mean = design_matrix(rows, periods_per_subject);
    let outcome = Mat::from_fn(rows, 1, |row, _| {
        let subject = row / periods_per_subject;
        let period = row % periods_per_subject;
        let exposure = if subject >= (subjects / 2) { 1.0 } else { 0.0 };
        let time_value = usize_to_f64(period);
        let subject_re = centered_subject_effect(subject, subjects);

        let linear_probability = 0.35f64.mul_add(
            time_value,
            0.9f64.mul_add(exposure, 0.7f64.mul_add(subject_re, -0.6)),
        );
        let probability_positive = logistic(linear_probability);
        let pseudo_uniform = usize_to_f64(row.saturating_mul(37).saturating_add(11) % 100) / 100.0;

        if pseudo_uniform <= probability_positive {
            let log_mean = 0.20f64.mul_add(
                time_value,
                0.35f64.mul_add(exposure, 0.3f64.mul_add(subject_re, 1.1)),
            );
            let deterministic_noise = 0.03f64.mul_add(usize_to_f64(row % 5), 1.0);
            log_mean.exp() * deterministic_noise
        } else {
            0.0
        }
    });

    LongitudinalModelInput::new(outcome, x_binary, x_mean, subject_ids, time)
}

fn design_matrix(rows: usize, periods_per_subject: usize) -> Mat<f64> {
    Mat::from_fn(rows, 3, |row, col| {
        let subject = row / periods_per_subject;
        match col {
            0 => 1.0,
            1 => usize_to_f64(row % periods_per_subject),
            _ => {
                if subject >= (rows / periods_per_subject / 2) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    })
}

fn build_counterfactual_scenario(periods: usize) -> CounterfactualScenario {
    let exposed_design = Mat::from_fn(
        periods,
        3,
        |row, col| if col == 1 { usize_to_f64(row) } else { 1.0 },
    );
    let unexposed_design = Mat::from_fn(periods, 3, |row, col| match col {
        0 => 1.0,
        1 => usize_to_f64(row),
        _ => 0.0,
    });

    CounterfactualScenario {
        binary_design_exposed: exposed_design.clone(),
        mean_design_exposed: exposed_design,
        binary_design_unexposed: unexposed_design.clone(),
        mean_design_unexposed: unexposed_design,
    }
}

fn centered_subject_effect(subject: usize, subjects: usize) -> f64 {
    let denominator = usize_to_f64(subjects).max(1.0);
    (usize_to_f64(subject) - denominator / 2.0) / denominator
}

fn logistic(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}
