//! The Tweedie fitter against R `glm(family = statmod::tweedie(var.power = p,
//! link.power = 0))` at powers 1.2, 1.5 and 1.8, on a design with exact zeros, with and
//! without prior weights. The fixture comes from `tests/r/tweedie_reference.R`.
//!
//! The crate reports no model-based covariance. Its standard errors are the sandwich
//! covariance with bread `X'diag(w mu^(2-p))X` and meat from the quasi-score
//! `w (y - mu) mu^(1-p) x`, so the reference standard errors are that same sandwich
//! computed in R from the glm fit's fitted means.

use faer::Mat;
use semicontinuous_models::{ModelInput, TweedieOptions, fit_tweedie_input};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    power: f64,
    weighted: bool,
    beta: Vec<f64>,
    se_robust: Vec<f64>,
    converged: bool,
}

#[derive(Deserialize)]
struct Fixture {
    x1: Vec<f64>,
    x2: Vec<f64>,
    y: Vec<f64>,
    weights: Vec<f64>,
    cases: Vec<Case>,
}

fn rel(got: f64, expected: f64) -> f64 {
    (got - expected).abs() / expected.abs().max(1e-12)
}

#[test]
fn tweedie_matches_statmod_glm() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/reference/tweedie_statmod.json"
    ))
    .expect("fixture");
    let fixture: Fixture = serde_json::from_str(&text).expect("fixture parses");
    let n = fixture.y.len();
    assert!(fixture.y.iter().filter(|&&v| v == 0.0).count() >= 8);
    assert_eq!(fixture.cases.len(), 6);

    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => fixture.x1[i],
        _ => fixture.x2[i],
    });
    let y = Mat::from_fn(n, 1, |i, _| fixture.y[i]);

    let mut worst_beta = 0.0_f64;
    let mut worst_se = 0.0_f64;
    for case in &fixture.cases {
        assert!(case.converged);
        let mut input = ModelInput::new(x.clone(), y.clone());
        if case.weighted {
            input = input.with_sample_weights(Mat::from_fn(n, 1, |i, _| fixture.weights[i]));
        }
        let options = TweedieOptions::builder()
            .max_iter(200_usize)
            .tolerance(1e-16_f64)
            .robust_se(true)
            .build();
        let (model, report) = fit_tweedie_input(&input, case.power, options).expect("tweedie fit");
        let se = report.se.expect("robust standard errors");
        for j in 0..3 {
            let beta_dev = rel(model.beta[(j, 0)], case.beta[j]);
            let se_dev = rel(se[(j, 0)], case.se_robust[j]);
            worst_beta = worst_beta.max(beta_dev);
            worst_se = worst_se.max(se_dev);
        }
    }
    eprintln!("worst relative deviation: beta {worst_beta:.3e}, se {worst_se:.3e}");
    // IRLS stops on a small step or a small relative change of the deviance, so the
    // coefficients agree with a fully converged glm to about 1e-8.
    assert!(worst_beta < 1e-7, "beta {worst_beta}");
    assert!(worst_se < 1e-7, "se {worst_se}");
}
