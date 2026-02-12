use faer::Mat;
use semicontinuous_models::{ModelInput, TweedieOptions, fit_tweedie_input};

fn main() {
    let n = 120;
    let design_matrix = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 25.0 });
    let outcome = Mat::from_fn(n, 1, |i, _| {
        if i % 6 == 0 {
            0.0
        } else {
            0.1f64.mul_add(idx_to_f64(i), 2.0)
        }
    });

    let input = ModelInput::new(design_matrix, outcome);
    let (model, _report) = fit_tweedie_input(&input, 1.5, TweedieOptions::default()).expect("fit");
    let pred = model.predict(&input.design_matrix);
    println!("first mean: {:.3}", pred.mean[(0, 0)]);
}

fn idx_to_f64(idx: usize) -> f64 {
    f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
}
