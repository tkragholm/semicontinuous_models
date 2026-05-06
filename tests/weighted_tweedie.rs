use approx::assert_relative_eq;
use faer::Mat;
use semicontinuous_models::{ModelInput, TweedieOptions, fit_tweedie_input};

#[test]
fn tweedie_input_uses_sample_weights_for_intercept_only_gamma_fit() {
    let x = Mat::from_fn(4, 1, |_row, _col| 1.0);
    let y = Mat::from_fn(4, 1, |row, _col| if row < 2 { 1.0 } else { 10.0 });
    let weights = Mat::from_fn(4, 1, |row, _col| if row < 2 { 10.0 } else { 1.0 });

    let weighted_mean = (10.0 * 1.0 + 10.0 * 1.0 + 1.0 * 10.0 + 1.0 * 10.0) / 22.0;
    let input = ModelInput::new(x, y).with_sample_weights(weights);

    let options = TweedieOptions::builder()
        .max_iter(100_usize)
        .tolerance(1e-10_f64)
        .build();
    let (model, _report) = fit_tweedie_input(&input, 2.0, options).expect("weighted gamma fit");

    assert_relative_eq!(model.beta[(0, 0)].exp(), weighted_mean, epsilon = 1e-6);
}
