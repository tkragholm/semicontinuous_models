//! Server-sized timing of the fits a clustered bootstrap runs per replicate.
//!
//! `cargo run --release --example fit_profile -- [n] [p] [replicates]`
//! Prints the median of three runs of each fit, one replicate at a time and
//! with replicates in parallel as the bootstrap driver runs them.
//!
//! `FWB=owned|borrowed cargo run --release --example fit_profile -- [n] [p] [replicates]`
//! runs only a fractional-weight bootstrap of the two-part model, the way the driver
//! does: every replicate fits the full design with its own weights and takes the
//! weighted mean of the prediction at two counterfactual designs. `owned` gives each
//! worker a copy of the design and fits and predicts through `ModelInput` and
//! `predict`. `borrowed` shares one design among the workers, fits through
//! `fit_two_part_ref_warm_prevalidated` and takes the means with `predict_mean`. Peak
//! memory is that of the process, so read it with `/usr/bin/time -l`.

use faer::Mat;
use rayon::prelude::*;
use semicontinuous_models::{
    FitOptions, Model, ModelInput, ModelInputRef, TweedieOptions, TwoPartWarmStart,
    fit_tweedie_input_warm, fit_two_part_input_warm, fit_two_part_ref_warm_prevalidated,
};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u = self.unif().max(1e-300);
        let v = self.unif();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Intercept, dummies of varying prevalence, and a few continuous columns.
pub fn make_data(n: usize, p: usize, zero_share: f64, seed: u64) -> ModelInput {
    let mut rng = Rng(seed | 1);
    let n_cont = 6.min(p.saturating_sub(1));
    let n_dummy = p - 1 - n_cont;
    let prev: Vec<f64> = (0..n_dummy)
        .map(|j| 0.03 + 0.4 * ((j * 7 % 13) as f64) / 13.0)
        .collect();
    let coef: Vec<f64> = (0..p)
        .map(|j| 0.15 * (((j * 5) % 7) as f64 - 3.0) / 3.0)
        .collect();
    let mut x = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        for j in 0..n_dummy {
            x[(i, 1 + j)] = f64::from(rng.unif() < prev[j]);
        }
        for j in 0..n_cont {
            x[(i, 1 + n_dummy + j)] = rng.normal();
        }
    }
    let mut y = Mat::<f64>::zeros(n, 1);
    let logit_shift = (zero_share / (1.0 - zero_share)).ln();
    for i in 0..n {
        let mut eta = 0.0;
        for j in 1..p {
            eta += coef[j] * x[(i, j)];
        }
        let p_pos = 1.0 / (1.0 + (logit_shift - eta * 0.8).exp());
        if rng.unif() < p_pos {
            // gamma(shape 2) via two exponentials, mean exp(eta + 8)
            let g = -(rng.unif().max(1e-300)).ln() - (rng.unif().max(1e-300)).ln();
            y[(i, 0)] = 0.5 * g * (8.0 + 0.5 * eta).exp();
        }
    }
    let mut input = ModelInput::new(x, y);
    input.sample_weights = Some(Mat::from_fn(n, 1, |_, _| 1.0));
    input
}

/// Row-resample (pairs bootstrap) as the driver builds a replicate.
fn resample(input: &ModelInput, seed: u64) -> ModelInput {
    let n = input.design_matrix.nrows();
    let mut rng = Rng(seed | 1);
    let rows: Vec<usize> = (0..n).map(|_| (rng.next() % n as u64) as usize).collect();
    let x = Mat::from_fn(n, input.design_matrix.ncols(), |i, j| {
        input.design_matrix[(rows[i], j)]
    });
    let y = Mat::from_fn(n, 1, |i, _| input.outcome[(rows[i], 0)]);
    let mut out = ModelInput::new(x, y);
    out.sample_weights = Some(Mat::from_fn(n, 1, |_, _| 1.0));
    out
}

fn fnv(mats: &[&Mat<f64>]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for m in mats {
        for i in 0..m.nrows() {
            for byte in m[(i, 0)].to_bits().to_le_bytes() {
                h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    h
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    eprintln!(
        "   (min {:.1}, median {:.1}, n={})",
        v[0],
        v[v.len() / 2],
        v.len()
    );
    v[v.len() / 2]
}

/// The design with column 1 set to `value`, as the counterfactual designs of a contrast.
fn with_column(x: &Mat<f64>, value: f64) -> Mat<f64> {
    let mut out = x.clone();
    for i in 0..out.nrows() {
        out[(i, 1)] = value;
    }
    out
}

/// Fractional-weight bootstrap replicates of the two-part model, owned or borrowed.
fn fwb(mode: &str, n: usize, p: usize, reps: usize, runs: usize) {
    let full = make_data(n, p, 0.4, 12345);
    let opts = FitOptions::stable_defaults();
    let (point, _) = fit_two_part_input_warm(&full, opts, None).expect("point fit");
    let warm = TwoPartWarmStart {
        beta_logit: point.beta_logit,
        beta_gamma: point.beta_gamma,
    };
    let ropts = FitOptions {
        compute_covariance: false,
        ..opts
    };
    let treated = with_column(&full.design_matrix, 1.0);
    let control = with_column(&full.design_matrix, 0.0);
    // The driver's averaging weights: the replicate weights on the exposed rows.
    let averaging: Vec<f64> = (0..n)
        .map(|i| {
            if full.design_matrix[(i, 1)] == 1.0 {
                1.0 + (i % 5) as f64
            } else {
                0.0
            }
        })
        .collect();
    let replicate_weights = |r: usize, out: &mut Mat<f64>| {
        let mut rng = Rng((5_000 + r as u64) | 1);
        for i in 0..n {
            out[(i, 0)] = 0.2 + 2.0 * rng.unif();
        }
    };
    let workers = rayon::current_num_threads();
    eprintln!("mode={mode} n={n} p={p} workers={workers} replicates={reps}");

    let run_one = |r: usize, state: &mut (Option<ModelInput>, Mat<f64>)| -> (f64, f64) {
        if mode == "owned" {
            let input = state.0.as_mut().expect("worker copy");
            replicate_weights(r, input.sample_weights.as_mut().expect("weights"));
            let (model, _) = fit_two_part_input_warm(input, ropts, Some(&warm)).expect("fit");
            let a = model.predict(&treated).expected_outcome;
            let b = model.predict(&control).expected_outcome;
            (mean(&a, Some(&averaging)), mean(&b, Some(&averaging)))
        } else {
            replicate_weights(r, &mut state.1);
            let input = ModelInputRef::new(full.design_matrix.as_ref(), full.outcome.as_ref())
                .with_sample_weights(state.1.as_ref());
            let (model, _) =
                fit_two_part_ref_warm_prevalidated(input, ropts, Some(&warm)).expect("fit");
            (
                model.predict_mean(&treated, Some(&averaging)),
                model.predict_mean(&control, Some(&averaging)),
            )
        }
    };
    let init = || {
        let copy = (mode == "owned").then(|| {
            let mut input = ModelInput::new(full.design_matrix.clone(), full.outcome.clone());
            input.sample_weights = Some(Mat::zeros(n, 1));
            input
        });
        (copy, Mat::<f64>::zeros(n, 1))
    };

    if mode == "borrowed" {
        // The two changes in isolation, one thread: the mean of the prediction at a
        // counterfactual design, and the validation of the design and outcome.
        let model = fit_two_part_input_warm(&full, ropts, None).expect("fit").0;
        let mut full_ms = Vec::new();
        let mut mean_ms = Vec::new();
        let mut validate_ms = Vec::new();
        for _ in 0..runs {
            let s = Instant::now();
            let a = mean(&model.predict(&treated).expected_outcome, Some(&averaging));
            full_ms.push(s.elapsed().as_secs_f64() * 1e3);
            let s = Instant::now();
            let b = model.predict_mean(&treated, Some(&averaging));
            mean_ms.push(s.elapsed().as_secs_f64() * 1e3);
            assert_eq!(a.to_bits(), b.to_bits());
            let s = Instant::now();
            full.as_ref().validate_core().expect("valid");
            validate_ms.push(s.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "predict + mean {:.2} ms, predict_mean {:.2} ms, validate_core {:.2} ms",
            median(full_ms),
            median(mean_ms),
            median(validate_ms)
        );
    }

    let mut t = Vec::new();
    let mut bits = 0_u64;
    for _ in 0..runs {
        let s = Instant::now();
        let results: Vec<(f64, f64)> = (0..reps)
            .into_par_iter()
            .map_init(init, |state, r| run_one(r, state))
            .collect();
        t.push(s.elapsed().as_secs_f64() * 1e3 / reps as f64);
        bits = results.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, (a, b)| {
            [a, b].iter().fold(h, |h, v| {
                v.to_bits().to_le_bytes().iter().fold(h, |h, &byte| {
                    (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
            })
        });
    }
    let mut state = init();
    let mut t1 = Vec::new();
    for _ in 0..runs {
        let s = Instant::now();
        run_one(0, &mut state);
        t1.push(s.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "{mode}: parallel {:.1} ms per replicate, single {:.1} ms, effects {bits:#018x}",
        median(t),
        median(t1)
    );
}

fn mean(values: &Mat<f64>, weights: Option<&[f64]>) -> f64 {
    let weights = weights.expect("weights");
    let (sum, total) = (0..values.nrows())
        .filter(|&i| weights[i] != 0.0)
        .fold((0.0, 0.0), |(sum, total), i| {
            (sum + weights[i] * values[(i, 0)], total + weights[i])
        });
    sum / total
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let n = args.first().copied().unwrap_or(234_000);
    let p = args.get(1).copied().unwrap_or(40);
    let reps = args.get(2).copied().unwrap_or(24);
    let runs: usize = std::env::var("RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    eprintln!("n={n} p={p} threads={}", rayon::current_num_threads());
    if let Ok(mode) = std::env::var("FWB") {
        fwb(&mode, n, p, reps, runs);
        return;
    }

    for (label, zero_share) in [("two-part (40% zeros)", 0.4)] {
        let full = make_data(n, p, zero_share, 12345);
        let zeros = (0..n).filter(|&i| full.outcome[(i, 0)] <= 0.0).count();
        eprintln!("{label}: zeros {:.1}%", 100.0 * zeros as f64 / n as f64);
        let opts = FitOptions::stable_defaults();
        let (point, rep) = fit_two_part_input_warm(&full, opts, None).expect("point fit");
        eprintln!(
            "point fit iterations logit {} gamma {}",
            rep.iterations_logit, rep.iterations_gamma
        );
        let warm = TwoPartWarmStart {
            beta_logit: point.beta_logit.clone(),
            beta_gamma: point.beta_gamma.clone(),
        };
        let ropts = FitOptions {
            compute_covariance: false,
            ..opts
        };
        let boots: Vec<ModelInput> = (0..reps)
            .map(|r| resample(&full, 1000 + r as u64))
            .collect();

        // one replicate at a time (inner parallelism only)
        let mut t = Vec::new();
        let mut its = (0, 0);
        let mut bits = 0;
        for _ in 0..runs {
            let s = Instant::now();
            let (m, r) = fit_two_part_input_warm(&boots[0], ropts, Some(&warm)).expect("fit");
            t.push(s.elapsed().as_secs_f64() * 1e3);
            its = (r.iterations_logit, r.iterations_gamma);
            bits = fnv(&[&m.beta_logit, &m.beta_gamma]);
        }
        println!(
            "two_part single replicate: {:.1} ms (iters logit {} gamma {}) betas {bits:#018x}",
            median(t),
            its.0,
            its.1
        );

        // all replicates in parallel, per-replicate amortised
        let mut t = Vec::new();
        for _ in 0..runs {
            let s = Instant::now();
            boots.par_iter().for_each(|b| {
                fit_two_part_input_warm(b, ropts, Some(&warm)).expect("fit");
            });
            t.push(s.elapsed().as_secs_f64() * 1e3 / reps as f64);
        }
        println!(
            "two_part parallel x{reps}: {:.1} ms per replicate",
            median(t)
        );

        // one-part gamma via Tweedie power 2, on the positive-only part is what the driver
        // fits to the full outcome (zeros included), so fit the full outcome here.
        let topts = TweedieOptions::default();
        let mut full = full;
        for i in 0..n {
            if full.outcome[(i, 0)] <= 0.0 {
                full.outcome[(i, 0)] = 1.0;
            }
        }
        let boots: Vec<ModelInput> = (0..reps)
            .map(|r| resample(&full, 1000 + r as u64))
            .collect();
        let (tpoint, trep) =
            fit_tweedie_input_warm(&full, 2.0, topts, None).expect("tweedie point");
        eprintln!("tweedie point iterations {}", trep.meta.iterations);
        let tw = tpoint.beta.clone();
        let mut t = Vec::new();
        let mut it = 0;
        let mut tbits = 0;
        for _ in 0..runs {
            let s = Instant::now();
            let (m, r) = fit_tweedie_input_warm(&boots[0], 2.0, topts, Some(&tw)).expect("fit");
            t.push(s.elapsed().as_secs_f64() * 1e3);
            it = r.meta.iterations;
            tbits = fnv(&[&m.beta]);
        }
        println!(
            "tweedie2 single replicate: {:.1} ms (iters {}) betas {tbits:#018x}",
            median(t),
            it
        );
        let mut t = Vec::new();
        for _ in 0..runs {
            let s = Instant::now();
            boots.par_iter().for_each(|b| {
                fit_tweedie_input_warm(b, 2.0, topts, Some(&tw)).expect("fit");
            });
            t.push(s.elapsed().as_secs_f64() * 1e3 / reps as f64);
        }
        println!(
            "tweedie2 parallel x{reps}: {:.1} ms per replicate",
            median(t)
        );
    }
}
