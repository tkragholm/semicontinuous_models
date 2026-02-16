use faer::Mat;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use semicontinuous_models::{
    CounterfactualScenario, LongitudinalModelInput, MtpFitOptions, MtpSamplerConfig,
    PositivePartDistribution, RandomEffectsStructure, compute_counterfactual_effects_summary,
    fit_mtp_input_with_posterior_config, posterior_predictive_summary,
};

const N_SUBJECTS: usize = 1200;
const N_PERIODS: usize = 5;
const TARGET_POPULATION: usize = 10_000;
const SIMULATION_SEED: u64 = 41;

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let truth = Truth {
        alpha: [-0.35, 0.30, -0.55],
        beta: [1.20, 0.20, -0.25],
        sigma: 0.35,
        re_sd_binary: 0.55,
        re_sd_mean: 0.40,
        re_correlation: 0.25,
    };
    let input = simulate_longitudinal_input(&truth, N_SUBJECTS, N_PERIODS, SIMULATION_SEED);

    let config = MtpSamplerConfig {
        fit_options: MtpFitOptions {
            iterations: 1_000,
            burn_in: 300,
            thin: 5,
            seed: 2_026,
            random_effects: RandomEffectsStructure::InterceptsOnly,
            ..MtpFitOptions::default()
        },
        positive_part_distribution: PositivePartDistribution::LogNormal,
        ..MtpSamplerConfig::default()
    };
    let (model, report, posterior) = fit_mtp_input_with_posterior_config(&input, config)?;

    println!("MTP policy-evaluation validation");
    println!("Scenario: preventive care program enrollment vs usual care.");
    println!("Outcome: quarterly non-drug healthcare costs with structural zeros.");
    println!(
        "Estimand: {N_PERIODS}-quarter incremental cost per enrolled participant and budget impact for {TARGET_POPULATION} people."
    );
    println!(
        "Synthetic cohort: n_subjects={}, periods={}, rows={}",
        N_SUBJECTS,
        N_PERIODS,
        input.outcome.nrows()
    );
    println!(
        "Model dimensions: binary covariates={}, mean covariates={}, random effect dim={}",
        model.n_binary_covariates, model.n_mean_covariates, model.random_effect_dimension
    );
    println!(
        "Sampler: iterations={}, retained draws={}",
        report.diagnostics.iterations_completed, report.diagnostics.retained_draws
    );

    let mut random_effect_acceptance = 0.0;
    if let Some(rates) = report.diagnostics.acceptance_rates {
        random_effect_acceptance = rates.random_effects;
        println!(
            "Acceptance rates: alpha={:.2}, beta={:.2}, random effects={:.2}, kappa={:.2}, omega_sq={:.2}",
            rates.alpha, rates.beta, rates.random_effects, rates.kappa, rates.omega_sq
        );
        if config.positive_part_distribution == PositivePartDistribution::LogNormal {
            println!("Note: kappa acceptance is expected to be 0.00 for LogNormal positive part.");
        }
    }

    let scenario = build_counterfactual_scenario(N_PERIODS);
    let effect_summary = compute_counterfactual_effects_summary(&posterior, &scenario)?;
    let ppc = posterior_predictive_summary(&posterior, &input, 10)?;

    let true_cumulative_effect = true_cumulative_additive_effect(&truth, N_PERIODS);
    let effect_relative_error = relative_error(
        effect_summary.cumulative_additive_effect.mean,
        true_cumulative_effect,
    );
    let zero_rate_error = (ppc.predicted_zero_rate.mean - ppc.observed_zero_rate).abs();
    let true_or_period0 = truth.alpha[2].exp();
    let estimated_or_period0 = effect_summary.per_period[0].odds_ratio_positive.mean;
    let covers_truth_interval = interval_contains(
        effect_summary.cumulative_additive_effect.q025,
        effect_summary.cumulative_additive_effect.q975,
        true_cumulative_effect,
    );
    let estimated_per_person_savings = -effect_summary.cumulative_additive_effect.mean;
    let estimated_savings_q025 = -effect_summary.cumulative_additive_effect.q975;
    let estimated_savings_q975 = -effect_summary.cumulative_additive_effect.q025;
    let estimated_population_savings =
        estimated_per_person_savings * usize_to_f64(TARGET_POPULATION);
    let estimated_population_savings_q025 =
        estimated_savings_q025 * usize_to_f64(TARGET_POPULATION);
    let estimated_population_savings_q975 =
        estimated_savings_q975 * usize_to_f64(TARGET_POPULATION);

    println!(
        "\nCounterfactual {}-quarter incremental cost (program - usual care): truth={:.3}, estimate={:.3}, 95% CrI [{:.3}, {:.3}]",
        N_PERIODS,
        true_cumulative_effect,
        effect_summary.cumulative_additive_effect.mean,
        effect_summary.cumulative_additive_effect.q025,
        effect_summary.cumulative_additive_effect.q975
    );
    println!(
        "Implied per-person {N_PERIODS}-quarter savings: estimate={estimated_per_person_savings:.3}, 95% CrI [{estimated_savings_q025:.3}, {estimated_savings_q975:.3}]"
    );
    println!(
        "Estimated total savings for {TARGET_POPULATION} participants: {estimated_population_savings:.1}, 95% CrI [{estimated_population_savings_q025:.1}, {estimated_population_savings_q975:.1}]"
    );
    println!(
        "Period 0 odds-ratio for any cost (program vs usual care): truth={true_or_period0:.3}, estimate={estimated_or_period0:.3}"
    );
    println!(
        "Posterior predictive: observed zero-rate={:.3}, predicted zero-rate={:.3}, |error|={:.3}, brier={:.3}",
        ppc.observed_zero_rate, ppc.predicted_zero_rate.mean, zero_rate_error, ppc.brier_score.mean
    );

    let checks = [
        (
            "random-effects acceptance > 0.05",
            random_effect_acceptance > 0.05,
        ),
        ("zero-rate error < 0.05", zero_rate_error < 0.05),
        (
            "counterfactual cumulative effect rel.error < 0.20",
            effect_relative_error < 0.20,
        ),
        (
            "cumulative effect interval covers truth",
            covers_truth_interval,
        ),
        (
            "period-0 odds-ratio direction recovered (<1)",
            estimated_or_period0 < 1.0,
        ),
        (
            "cost-saving sign recovered (incremental cost < 0)",
            effect_summary.cumulative_additive_effect.mean < 0.0,
        ),
    ];

    println!("\nValidation checks");
    let passed = checks.iter().filter(|(_, ok)| *ok).count();
    for (name, ok) in checks {
        let status = if ok { "PASS" } else { "FAIL" };
        println!("{name}: {status}");
    }
    println!("Checks passed: {passed}/6");

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct Truth {
    alpha: [f64; 3],
    beta: [f64; 3],
    sigma: f64,
    re_sd_binary: f64,
    re_sd_mean: f64,
    re_correlation: f64,
}

fn simulate_longitudinal_input(
    truth: &Truth,
    subjects: usize,
    periods_per_subject: usize,
    seed: u64,
) -> LongitudinalModelInput {
    let rows = subjects.saturating_mul(periods_per_subject);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut subject_re_binary = vec![0.0; subjects];
    let mut subject_re_mean = vec![0.0; subjects];
    let mut exposure_by_subject = vec![0.0; subjects];

    for subject in 0..subjects {
        let z1 = sample_standard_normal(&mut rng);
        let z2 = sample_standard_normal(&mut rng);
        let rho = truth.re_correlation.clamp(-0.99, 0.99);
        let z_corr = rho.mul_add(z1, (1.0 - rho * rho).sqrt() * z2);

        subject_re_binary[subject] = truth.re_sd_binary * z1;
        subject_re_mean[subject] = truth.re_sd_mean * z_corr;
        exposure_by_subject[subject] = if rng.random::<f64>() < 0.5 { 1.0 } else { 0.0 };
    }

    let subject_ids = (0..rows)
        .map(|row| u64::try_from(row / periods_per_subject).unwrap_or(u64::MAX) + 1)
        .collect::<Vec<_>>();
    let time = (0..rows)
        .map(|row| usize_to_f64(row % periods_per_subject))
        .collect::<Vec<_>>();
    let x_binary = design_matrix(rows, periods_per_subject, &exposure_by_subject);
    let x_mean = design_matrix(rows, periods_per_subject, &exposure_by_subject);

    let outcome = Mat::from_fn(rows, 1, |row, _| {
        let subject = row / periods_per_subject;
        let period = row % periods_per_subject;
        let exposure = exposure_by_subject[subject];
        let time_value = usize_to_f64(period);

        let eta_binary = truth.alpha[1].mul_add(
            time_value,
            truth.alpha[2].mul_add(exposure, truth.alpha[0] + subject_re_binary[subject]),
        );
        let p_positive = logistic(eta_binary);
        if rng.random::<f64>() >= p_positive {
            return 0.0;
        }

        let eta_mean = truth.beta[1].mul_add(
            time_value,
            truth.beta[2].mul_add(exposure, truth.beta[0] + subject_re_mean[subject]),
        );
        let epsilon = sample_standard_normal(&mut rng);
        (truth.sigma.mul_add(epsilon, eta_mean)).exp()
    });

    LongitudinalModelInput::new(outcome, x_binary, x_mean, subject_ids, time)
}

fn design_matrix(rows: usize, periods_per_subject: usize, exposure_by_subject: &[f64]) -> Mat<f64> {
    Mat::from_fn(rows, 3, |row, col| {
        let subject = row / periods_per_subject;
        match col {
            0 => 1.0,
            1 => usize_to_f64(row % periods_per_subject),
            _ => exposure_by_subject[subject],
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

fn true_cumulative_additive_effect(truth: &Truth, periods: usize) -> f64 {
    let mut cumulative = 0.0;
    for period in 0..periods {
        let time = usize_to_f64(period);
        let exposed_mean = truth.beta[1].mul_add(time, truth.beta[2].mul_add(1.0, truth.beta[0]));
        let unexposed_mean = truth.beta[1].mul_add(time, truth.beta[2].mul_add(0.0, truth.beta[0]));
        cumulative += exposed_mean.exp() - unexposed_mean.exp();
    }
    cumulative
}

fn interval_contains(lower: f64, upper: f64, value: f64) -> bool {
    lower <= value && value <= upper
}

fn relative_error(estimate: f64, truth: f64) -> f64 {
    ((estimate - truth) / truth).abs()
}

fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = (1.0_f64 - rng.random::<f64>()).max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0_f64 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn logistic(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}
