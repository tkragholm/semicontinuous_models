//! Fits with `inner_parallel: false` run every inner pass on the calling thread.
//!
//! The serial results are pinned as bit patterns, must not depend on the rayon thread
//! count or on `RAYON_NUM_THREADS`, and must stay close to the parallel results.

use faer::Mat;
use semicontinuous_models::{
    FitOptions, Model, ModelInput, TweedieOptions, TwoPartWarmStart, fit_tweedie_input_warm,
    fit_two_part_input_warm,
};
use std::process::Command;

const N_ROWS: usize = 20_000;
const N_COLS: usize = 12;
const THREAD_COUNTS: [usize; 4] = [1, 2, 3, 12];

struct Stream(u64);

impl Stream {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// Intercept, eight dummies, three continuous columns; about 40% zeros when `zeros`
/// is set; positive weights and family-sized clusters.
fn fixture(zeros: bool) -> ModelInput {
    let mut stream = Stream(20_261_029);
    let mut x = Mat::<f64>::zeros(N_ROWS, N_COLS);
    for i in 0..N_ROWS {
        x[(i, 0)] = 1.0;
        for j in 1..9 {
            let prevalence = 0.05 + 0.05 * j as f64;
            x[(i, j)] = f64::from(stream.uniform() < prevalence);
        }
        for j in 9..N_COLS {
            x[(i, j)] = 2.0f64.mul_add(stream.uniform(), -1.0);
        }
    }
    let mut y = Mat::<f64>::zeros(N_ROWS, 1);
    for i in 0..N_ROWS {
        let eta = 0.4f64.mul_add(
            x[(i, 1)],
            0.2 * x[(i, 2)] - 0.1 * x[(i, 9)] + 0.15 * x[(i, 10)],
        );
        let is_zero = zeros && stream.uniform() < 0.4 - 0.1 * x[(i, 3)];
        let noise = -stream.uniform().max(1e-12).ln() - stream.uniform().max(1e-12).ln();
        if !is_zero {
            y[(i, 0)] = (8.0f64 + eta).exp() / 2.0 * noise + 1.0;
        }
    }
    let mut clusters = Vec::with_capacity(N_ROWS);
    let mut cluster = 0_u64;
    while clusters.len() < N_ROWS {
        let size = 1 + (stream.next_u64() % 4) as usize;
        for _ in 0..size.min(N_ROWS - clusters.len()) {
            clusters.push(cluster * 7_919 + 13);
        }
        cluster += 1;
    }
    ModelInput::new(x, y)
        .with_cluster_ids(clusters)
        .with_sample_weights(Mat::from_fn(N_ROWS, 1, |_, _| 0.5 + stream.uniform()))
}

/// Coefficients, standard errors and predictions of one fit, in that order.
struct Fit {
    name: &'static str,
    coefficients: Vec<f64>,
    standard_errors: Vec<f64>,
    predictions: Vec<f64>,
}

fn flatten(matrices: &[&Mat<f64>]) -> Vec<f64> {
    matrices
        .iter()
        .flat_map(|m| (0..m.nrows()).map(move |i| (0..m.ncols()).map(move |j| m[(i, j)])))
        .flatten()
        .collect()
}

fn collect(inner_parallel: bool) -> Vec<Fit> {
    let zeros = fixture(true);
    let positive = fixture(false);
    let mut out = Vec::new();

    let options = FitOptions {
        robust_se: true,
        inner_parallel,
        ..FitOptions::default()
    };
    let (model, report) = fit_two_part_input_warm(&zeros, options, None).expect("two-part");
    let prediction = model.predict(&zeros.design_matrix);
    let ses = [
        report.se_logit.as_ref().expect("se"),
        report.se_gamma.as_ref().expect("se"),
    ];
    out.push(Fit {
        name: "two_part/robust/clustered",
        coefficients: flatten(&[&model.beta_logit, &model.beta_gamma]),
        standard_errors: flatten(&ses),
        predictions: flatten(&[&prediction.expected_outcome]),
    });
    let warm = TwoPartWarmStart {
        beta_logit: model.beta_logit,
        beta_gamma: model.beta_gamma,
    };

    let options = FitOptions {
        inner_parallel,
        ..FitOptions::default()
    };
    let (model, report) = fit_two_part_input_warm(&zeros, options, None).expect("two-part");
    let prediction = model.predict(&zeros.design_matrix);
    out.push(Fit {
        name: "two_part/model_se",
        coefficients: flatten(&[&model.beta_logit, &model.beta_gamma]),
        standard_errors: flatten(&[
            report.se_logit.as_ref().expect("se"),
            report.se_gamma.as_ref().expect("se"),
        ]),
        predictions: flatten(&[&prediction.expected_outcome]),
    });

    let options = FitOptions {
        compute_covariance: false,
        inner_parallel,
        ..FitOptions::stable_defaults()
    };
    let (model, _) = fit_two_part_input_warm(&zeros, options, Some(&warm)).expect("two-part");
    let prediction = model.predict(&zeros.design_matrix);
    out.push(Fit {
        name: "two_part/warm/no_cov",
        coefficients: flatten(&[&model.beta_logit, &model.beta_gamma]),
        standard_errors: Vec::new(),
        predictions: flatten(&[&prediction.expected_outcome]),
    });

    for (name, power, input) in [
        ("tweedie2/robust/clustered", 2.0, &positive),
        ("tweedie1.5/robust/clustered", 1.5, &zeros),
    ] {
        let options = TweedieOptions {
            robust_se: true,
            inner_parallel,
            ..TweedieOptions::default()
        };
        let (model, report) = fit_tweedie_input_warm(input, power, options, None).expect(name);
        out.push(Fit {
            name,
            coefficients: flatten(&[&model.beta]),
            standard_errors: flatten(&[report.se.as_ref().expect("se")]),
            predictions: flatten(&[&model.predict(&input.design_matrix).mean]),
        });
    }
    out
}

fn hash(fits: &[Fit]) -> Vec<(&'static str, u64)> {
    fits.iter()
        .map(|fit| {
            let mut h = 0xcbf2_9ce4_8422_2325_u64;
            for value in fit
                .coefficients
                .iter()
                .chain(&fit.standard_errors)
                .chain(&fit.predictions)
            {
                for byte in value.to_bits().to_le_bytes() {
                    h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            (fit.name, h)
        })
        .collect()
}

fn largest_differences(a: &[f64], b: &[f64]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold((0.0, 0.0), |(abs, rel), (x, y)| {
        let diff = (x - y).abs();
        (
            f64::max(abs, diff),
            f64::max(rel, diff / x.abs().max(y.abs()).max(f64::MIN_POSITIVE)),
        )
    })
}

#[test]
fn serial_fits_do_not_depend_on_the_rayon_thread_count() {
    let reference = hash(&collect(false));
    for threads in THREAD_COUNTS {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool");
        let hashes = pool.install(|| hash(&collect(false)));
        assert_eq!(hashes, reference, "{threads} threads differs");
    }
}

/// Prints the hashes; the parent test runs it as a child process under different
/// `RAYON_NUM_THREADS` values.
#[test]
#[ignore = "run by serial_fits_do_not_depend_on_rayon_num_threads"]
fn serial_hashes_child() {
    for (name, hash) in hash(&collect(false)) {
        println!("HASH {name} {hash:#018x}");
    }
}

#[test]
fn serial_fits_do_not_depend_on_rayon_num_threads() {
    let exe = std::env::current_exe().expect("test binary");
    let run = |threads: &str| -> String {
        let output = Command::new(&exe)
            .args(["--ignored", "--exact", "serial_hashes_child", "--nocapture"])
            .env("RAYON_NUM_THREADS", threads)
            .output()
            .expect("child test");
        assert!(output.status.success(), "child failed at {threads} threads");
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|line| line.starts_with("HASH "))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let reference = run("1");
    assert!(reference.lines().count() >= 5, "child printed no hashes");
    for threads in ["2", "5", "12"] {
        assert_eq!(
            run(threads),
            reference,
            "RAYON_NUM_THREADS={threads} differs"
        );
    }
    let in_process = hash(&collect(false))
        .iter()
        .map(|(name, h)| format!("HASH {name} {h:#018x}"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(in_process, reference, "child and parent differ");
}

#[test]
fn serial_fits_stay_close_to_the_parallel_fits() {
    let serial = collect(false);
    let parallel = collect(true);
    for (s, p) in serial.iter().zip(&parallel) {
        for (what, a, b) in [
            ("coefficients", &s.coefficients, &p.coefficients),
            ("standard errors", &s.standard_errors, &p.standard_errors),
            ("predictions", &s.predictions, &p.predictions),
        ] {
            let (abs, rel) = largest_differences(a, b);
            println!("{} {what}: max abs {abs:.3e}, max rel {rel:.3e}", s.name);
            assert!(rel < 1e-9, "{} {what}: relative difference {rel:e}", s.name);
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[test]
fn serial_fits_match_the_recorded_bit_patterns() {
    const GOLDEN: &[(&str, u64)] = &[
        ("two_part/robust/clustered", 0x85d6_4422_4881_372a),
        ("two_part/model_se", 0xccd5_d65d_7055_f000),
        ("two_part/warm/no_cov", 0xa99b_e55f_83d3_9adf),
        ("tweedie2/robust/clustered", 0xb42e_4db2_4504_51b4),
        ("tweedie1.5/robust/clustered", 0x28a2_0d9c_3320_38af),
    ];
    let actual = hash(&collect(false));
    if std::env::var_os("GOLDEN_PRINT").is_some() {
        for (name, hash) in &actual {
            println!("    (\"{name}\", {hash:#018x}),");
        }
    }
    assert_eq!(actual.len(), GOLDEN.len(), "golden table is out of date");
    for ((name, hash), (golden_name, golden_hash)) in actual.iter().zip(GOLDEN) {
        assert_eq!(name, golden_name);
        assert_eq!(*hash, *golden_hash, "{name}: output bits changed");
    }
}
