//! Golden outputs of the fitters, pinned as exact f64 bit patterns.
//!
//! Each entry is an FNV-1a hash over the bit patterns of every coefficient,
//! standard error, covariance entry or prediction a fit returns, plus the
//! iteration counts. Any change to the order of floating-point operations in
//! the fitters changes a hash.
//!
//! The hashes were recorded on aarch64 macOS. The libm `exp` and `ln` and the
//! gemm kernels faer selects differ between CPUs, so the test is compiled only
//! on that target. Regenerate with `GOLDEN_PRINT=1 cargo test --test fit_golden
//! -- --nocapture` after a change that is meant to alter results.
#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use faer::Mat;
use semicontinuous_models::{
    FitOptions, LogNormalOptions, Model, ModelInput, TweedieOptions, TwoPartWarmStart,
    fit_lognormal_smearing_input, fit_tweedie_input, fit_tweedie_input_warm, fit_two_part_input,
    fit_two_part_input_warm,
};

const N_ROWS: usize = 10_000;
const N_COLS: usize = 12;

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

/// Intercept, eight dummies, three continuous columns; about 40% zeros when
/// `zeros` is set; positive weights and family-sized clusters.
fn fixture(zeros: bool, weighted: bool, clustered: bool) -> ModelInput {
    let mut stream = Stream(20_261_028);
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
        let eta = 0.4f64.mul_add(x[(i, 1)], 0.2 * x[(i, 2)] - 0.1 * x[(i, 9)] + 0.15 * x[(i, 10)]);
        let is_zero = zeros && stream.uniform() < 0.4 - 0.1 * x[(i, 3)];
        let noise = -stream.uniform().max(1e-12).ln() - stream.uniform().max(1e-12).ln();
        if !is_zero {
            y[(i, 0)] = (8.0f64 + eta).exp() / 2.0 * noise + 1.0;
        }
    }
    let mut input = ModelInput::new(x, y);
    if clustered {
        let mut clusters = Vec::with_capacity(N_ROWS);
        let mut cluster = 0_u64;
        while clusters.len() < N_ROWS {
            let size = 1 + (stream.next_u64() % 4) as usize;
            for _ in 0..size.min(N_ROWS - clusters.len()) {
                clusters.push(cluster * 7_919 + 13);
            }
            cluster += 1;
        }
        input = input.with_cluster_ids(clusters);
    }
    if weighted {
        input = input.with_sample_weights(Mat::from_fn(N_ROWS, 1, |_, _| 0.5 + stream.uniform()));
    }
    input
}

struct Hasher(u64);

impl Hasher {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn word(&mut self, word: u64) {
        for byte in word.to_le_bytes() {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn mat(&mut self, m: &Mat<f64>) {
        self.word(m.nrows() as u64);
        self.word(m.ncols() as u64);
        for j in 0..m.ncols() {
            for i in 0..m.nrows() {
                self.word(m[(i, j)].to_bits());
            }
        }
    }

    fn opt(&mut self, m: &Option<Mat<f64>>) {
        match m {
            Some(m) => self.mat(m),
            None => self.word(u64::MAX),
        }
    }
}

type Golden = Vec<(&'static str, u64)>;

fn two_part(
    out: &mut Golden,
    name: &'static str,
    input: &ModelInput,
    options: FitOptions,
    warm: Option<&TwoPartWarmStart>,
) -> TwoPartWarmStart {
    let (model, report) = fit_two_part_input_warm(input, options, warm).expect(name);
    let mut h = Hasher::new();
    h.mat(&model.beta_logit);
    h.mat(&model.beta_gamma);
    h.opt(&report.se_logit);
    h.opt(&report.se_gamma);
    h.opt(&report.cov_logit);
    h.opt(&report.cov_gamma);
    h.word(report.iterations_logit as u64);
    h.word(report.iterations_gamma as u64);
    h.word(report.meta.fallback_attempts as u64);
    let prediction = model.predict(&input.design_matrix);
    h.mat(&prediction.prob_positive);
    h.mat(&prediction.mean_positive);
    h.mat(&prediction.expected_outcome);
    out.push((name, h.0));
    TwoPartWarmStart {
        beta_logit: model.beta_logit,
        beta_gamma: model.beta_gamma,
    }
}

fn tweedie(
    out: &mut Golden,
    name: &'static str,
    input: &ModelInput,
    power: f64,
    options: TweedieOptions,
    warm: Option<&Mat<f64>>,
) -> Mat<f64> {
    let (model, report) = fit_tweedie_input_warm(input, power, options, warm).expect(name);
    let mut h = Hasher::new();
    h.mat(&model.beta);
    h.opt(&report.se);
    h.opt(&report.cov);
    h.word(report.meta.iterations as u64);
    h.word(report.meta.fallback_attempts as u64);
    h.mat(&model.predict(&input.design_matrix).mean);
    out.push((name, h.0));
    model.beta
}

fn collect() -> Golden {
    let mut out = Golden::new();

    // Two-part: robust clustered SEs, weighted and unweighted, plain model-based SEs,
    // ridge with the relaxed strategy, and bootstrap-style warm starts.
    let zeros_unweighted = fixture(true, false, true);
    let zeros_weighted = fixture(true, true, true);
    let zeros_plain = fixture(true, true, false);
    two_part(
        &mut out,
        "two_part/unweighted/robust/clustered",
        &zeros_unweighted,
        FitOptions::builder().robust_se(true).build(),
        None,
    );
    two_part(
        &mut out,
        "two_part/unweighted/model_se",
        &fixture(true, false, false),
        FitOptions::default(),
        None,
    );
    two_part(
        &mut out,
        "two_part/weighted/model_se",
        &zeros_plain,
        FitOptions::default(),
        None,
    );
    let warm = two_part(
        &mut out,
        "two_part/weighted/robust/clustered",
        &zeros_weighted,
        FitOptions::builder().robust_se(true).build(),
        None,
    );
    let stable = FitOptions::stable_defaults();
    two_part(
        &mut out,
        "two_part/weighted/stable",
        &zeros_weighted,
        stable,
        None,
    );
    let replicate_options = FitOptions {
        compute_covariance: false,
        ..stable
    };
    two_part(
        &mut out,
        "two_part/weighted/warm/no_cov",
        &zeros_plain,
        replicate_options,
        Some(&warm),
    );
    two_part(
        &mut out,
        "two_part/weighted/warm/model_se",
        &zeros_plain,
        FitOptions::default(),
        Some(&warm),
    );

    // Tweedie: gamma (power 2) with cached factor, power 1.5 with per-iteration
    // weights, robust and not, warm and cold.
    let positive_weighted = fixture(false, true, true);
    let positive_plain = fixture(false, false, false);
    let robust = TweedieOptions::builder().robust_se(true).build();
    let cold = tweedie(
        &mut out,
        "tweedie2/weighted/robust/clustered",
        &positive_weighted,
        2.0,
        robust,
        None,
    );
    tweedie(
        &mut out,
        "tweedie2/unweighted/plain",
        &positive_plain,
        2.0,
        TweedieOptions::default(),
        None,
    );
    tweedie(
        &mut out,
        "tweedie2/weighted/warm",
        &positive_weighted,
        2.0,
        TweedieOptions::default(),
        Some(&cold),
    );
    tweedie(
        &mut out,
        "tweedie2/weighted/ridge/warm/robust",
        &positive_weighted,
        2.0,
        TweedieOptions::builder().l2_penalty(1e-4).robust_se(true).build(),
        Some(&cold),
    );
    tweedie(
        &mut out,
        "tweedie1.5/weighted/robust/clustered",
        &fixture(true, true, true),
        1.5,
        robust,
        None,
    );
    tweedie(
        &mut out,
        "tweedie1.5/unweighted/plain",
        &fixture(true, false, false),
        1.5,
        TweedieOptions::default(),
        None,
    );

    // Log-normal, which shares the Gram and matrix-vector helpers.
    let (model, report) = fit_lognormal_smearing_input(
        &fixture(true, true, true),
        LogNormalOptions::builder().robust_se(true).build(),
    )
    .expect("lognormal");
    let mut h = Hasher::new();
    h.mat(&model.beta);
    h.opt(&report.se);
    h.opt(&report.cov);
    out.push(("lognormal/weighted/robust/clustered", h.0));

    // The plain-API wrappers used elsewhere.
    let (model, _) = fit_two_part_input(&zeros_plain, FitOptions::default()).expect("two-part");
    let mut h = Hasher::new();
    h.mat(&model.beta_logit);
    h.mat(&model.beta_gamma);
    out.push(("two_part/input/default", h.0));
    let (model, _) =
        fit_tweedie_input(&positive_plain, 2.0, TweedieOptions::default()).expect("tweedie");
    let mut h = Hasher::new();
    h.mat(&model.beta);
    out.push(("tweedie2/input/default", h.0));

    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("two_part/unweighted/robust/clustered", 0x3e3d9fae257d710a),
    ("two_part/unweighted/model_se", 0x7238922a71787077),
    ("two_part/weighted/model_se", 0x288fd34eb86d1456),
    ("two_part/weighted/robust/clustered", 0x72815ba62317b4af),
    ("two_part/weighted/stable", 0x18cb2aa04df1f58e),
    ("two_part/weighted/warm/no_cov", 0xc925fead5cba18f0),
    ("two_part/weighted/warm/model_se", 0xa90112d7dbe6f2b5),
    ("tweedie2/weighted/robust/clustered", 0x45c26879984fe036),
    ("tweedie2/unweighted/plain", 0x66ca3ead887a50d8),
    ("tweedie2/weighted/warm", 0x169946344253d936),
    ("tweedie2/weighted/ridge/warm/robust", 0xdef64d567232e1b3),
    ("tweedie1.5/weighted/robust/clustered", 0xcd192701a061b424),
    ("tweedie1.5/unweighted/plain", 0x7366d249674b0c28),
    ("lognormal/weighted/robust/clustered", 0x32c2c90488bab51a),
    ("two_part/input/default", 0x50443c256527acf3),
    ("tweedie2/input/default", 0xfeca4cccd30d521c),
];

#[test]
fn fits_match_the_recorded_bit_patterns() {
    let actual = collect();
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
