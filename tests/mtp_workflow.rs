use faer::Mat;
use semicontinuous_models::{
    CounterfactualScenario, LongitudinalModelInput, MtpFitOptions, RandomEffectsStructure,
    compute_counterfactual_effects_summary, fit_mtp_input_with_posterior,
    posterior_predictive_summary,
};

fn idx_to_f64(idx: usize) -> f64 {
    f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
}

fn sample_input() -> LongitudinalModelInput {
    LongitudinalModelInput::new(
        Mat::from_fn(20, 1, |row, _| {
            if row % 2 == 0 {
                0.0
            } else {
                1.0 + idx_to_f64(row % 5)
            }
        }),
        Mat::from_fn(20, 1, |_row, _col| 1.0),
        Mat::from_fn(20, 1, |_row, _col| 1.0),
        vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
        vec![
            0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0,
            1.0, 2.0, 3.0,
        ],
    )
}

#[test]
fn mtp_public_workflow_produces_effects_and_ppc() {
    let input = sample_input();
    let options = MtpFitOptions {
        iterations: 160,
        burn_in: 40,
        thin: 4,
        seed: 123,
        random_effects: RandomEffectsStructure::InterceptsOnly,
        adapt_during_burn_in: true,
    };

    let (_model, report, samples) =
        fit_mtp_input_with_posterior(&input, options).expect("fit should succeed");

    assert_eq!(samples.len(), report.diagnostics.retained_draws);
    assert!(report.diagnostics.acceptance_rates.is_some());
    assert!(report.posterior_summary.is_some());

    let scenario = CounterfactualScenario {
        binary_design_exposed: Mat::from_fn(4, 1, |_row, _col| 1.0),
        mean_design_exposed: Mat::from_fn(4, 1, |_row, _col| 1.0),
        binary_design_unexposed: Mat::from_fn(4, 1, |_row, _col| 1.0),
        mean_design_unexposed: Mat::from_fn(4, 1, |_row, _col| 1.0),
    };

    let effects =
        compute_counterfactual_effects_summary(&samples, &scenario).expect("effects should run");
    assert_eq!(effects.per_period.len(), 4);
    assert!(effects.cumulative_additive_effect.mean.is_finite());

    let ppc = posterior_predictive_summary(&samples, &input, 8).expect("ppc should run");
    assert_eq!(ppc.calibration_bins.len(), 8);
    assert!(ppc.predicted_mean.mean.is_finite());
    assert!(ppc.predicted_zero_rate.mean >= 0.0);
}
