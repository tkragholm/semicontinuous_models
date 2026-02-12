use faer::Mat;
use semicontinuous_models::{
    FitOptions, ModelComparisonOptions, ModelInput, Regularization, compare_models_input,
    render_comparison_tables,
};

fn main() {
    let input = build_sample_input(200);
    let options = ModelComparisonOptions {
        two_part_elastic_net_options: Some(FitOptions {
            regularization: Regularization::ElasticNet {
                lambda: 0.1,
                alpha: 0.5,
                exclude_intercept: true,
            },
            ..FitOptions::default()
        }),
        ..ModelComparisonOptions::default()
    };
    let report = compare_models_input(&input, &options).expect("compare models");

    let tables = render_comparison_tables(&report);
    println!(
        "Model comparison (lower is better; R2 higher is better)\n\n{}",
        tables.in_sample
    );
    println!(
        "\nInformation criteria (quasi-likelihood for Tweedie)\n\n{}",
        tables.information_criteria
    );
    println!(
        "\nTweedie candidate information criteria (quasi-likelihood)\n\n{}",
        tables.tweedie_candidates
    );
    println!(
        "\n5-fold CV (expected outcome metrics)\n\n{}",
        tables.cv_summary
    );
    println!("\nCV ranking (by RMSE)\n\n{}", tables.cv_ranking);
    println!(
        "\nTweedie CV ranking (by RMSE) + in-sample AIC/BIC\n\n{}",
        tables.tweedie_cv_ranking
    );
}

fn build_sample_input(n: usize) -> ModelInput {
    let design_matrix = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 50.0 });
    let outcome = Mat::from_fn(n, 1, |i, _| {
        if i % 4 == 0 {
            0.0
        } else {
            0.2f64.mul_add(idx_to_f64(i), 1.0)
        }
    });
    ModelInput::new(design_matrix, outcome)
}

fn idx_to_f64(idx: usize) -> f64 {
    f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
}
