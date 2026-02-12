use faer::Mat;
use semicontinuous_models::{
    BootstrapOptions, FitOptions, ModelInput, bootstrap_two_part, fit_two_part_input,
};

fn main() {
    let n = 200;
    let design_matrix = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 50.0 });
    let outcome = Mat::from_fn(n, 1, |i, _| {
        if i % 4 == 0 {
            0.0
        } else {
            0.2f64.mul_add(idx_to_f64(i), 1.0)
        }
    });

    let input = ModelInput::new(design_matrix, outcome);
    let (model, report) = fit_two_part_input(&input, FitOptions::default()).expect("fit");
    println!(
        "logit iters: {}, gamma iters: {}",
        report.iterations_logit, report.iterations_gamma
    );

    let bootstrap = BootstrapOptions {
        iterations: 50,
        seed: 7,
        ..BootstrapOptions::default()
    };
    let boot = bootstrap_two_part(
        &input.design_matrix,
        &input.outcome,
        FitOptions::default(),
        bootstrap,
    )
    .expect("bootstrap");
    println!("bootstrap draws: {}", boot.betas_logit.len());

    let preds = model.predict(&input.design_matrix);
    println!(
        "first expected outcome: {:.3}",
        preds.expected_outcome[(0, 0)]
    );
}

fn idx_to_f64(idx: usize) -> f64 {
    f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
}
