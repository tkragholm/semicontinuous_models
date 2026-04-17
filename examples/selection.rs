use faer::Mat;
use semicontinuous_models::{
    CrossValidationOptions, ModelInput, cross_validate_models_input, recommend_from_cv,
    select_models_input,
};

fn main() {
    let n = 100;
    let design_matrix = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 20.0 });
    let outcome = Mat::from_fn(n, 1, |i, _| {
        if i % 5 == 0 {
            0.0
        } else {
            0.15f64.mul_add(idx_to_f64(i), 1.5)
        }
    });

    let input = ModelInput::new(design_matrix, outcome);
    let result = select_models_input(&input, &[1.2, 1.5, 1.8]);
    println!("Park slope: {:.3}", result.park_test.slope);
    println!("Tweedie candidates: {}", result.tweedie_candidates.len());
    for candidate in &result.tweedie_candidates {
        println!(
            "Tweedie p={:.2}: rmse {:.4}, mae {:.4}, rmsle {:.4}, r2 {:.4}, dev {:.4}",
            candidate.power,
            candidate.metrics.rmse,
            candidate.metrics.mae,
            candidate.metrics.rmsle,
            candidate.metrics.r2,
            candidate.metrics.deviance
        );
    }
    if let Some(lognormal) = result.lognormal_candidate {
        println!(
            "Lognormal: rmse {:.4}, mae {:.4}, rmsle {:.4}, r2 {:.4}, dev {:.4}",
            lognormal.metrics.rmse,
            lognormal.metrics.mae,
            lognormal.metrics.rmsle,
            lognormal.metrics.r2,
            lognormal.metrics.deviance
        );
    }

    if let Some(recommended) = result.recommended_by_aic {
        println!("Recommended by AIC: {recommended}");
    }
    if let Some(recommended) = result.recommended_by_bic {
        println!("Recommended by BIC: {recommended}");
    }

    let cv =
        cross_validate_models_input(&input, &[1.2, 1.5, 1.8], CrossValidationOptions::default())
            .expect("cv");
    if let Some(recommended) = recommend_from_cv(&cv) {
        println!("Recommended by CV RMSE: {recommended}");
    }
}

fn idx_to_f64(idx: usize) -> f64 {
    f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
}
