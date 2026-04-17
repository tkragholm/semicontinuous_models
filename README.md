# semicontinuous_models

`semicontinuous_models` provides reusable model implementations for semi-continuous outcomes:

- Two-part model (logit + gamma log link)
- Tweedie GLM (quasi-likelihood)
- Log-normal regression with smearing retransformation
- Selection and comparison utilities (Park test, AIC/BIC, cross-validation)

The crate was originally developed for a healthcare outcome study, but the API is domain-agnostic and can be reused for any non-negative semi-continuous response.

## Installation

```toml
[dependencies]
semicontinuous_models = "0.1"
```

## Quick start

```rust
use semicontinuous_models::{FitOptions, ModelInput, fit_two_part_input};
use faer::Mat;

fn idx_to_f64(idx: usize) -> f64 {
    f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
}

let n = 40;
let design_matrix = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { idx_to_f64(i) / 10.0 });
let outcome = Mat::from_fn(n, 1, |i, _| if i % 4 == 0 { 0.0 } else { 1.0 + 0.1 * idx_to_f64(i) });
let input = ModelInput::new(design_matrix, outcome);

let (model, report) = fit_two_part_input(&input, FitOptions::default()).expect("fit");
let prediction = model.predict(&input.design_matrix);

assert_eq!(prediction.expected_outcome.nrows(), n);
assert!(report.iterations_logit > 0);
```

For more numerically stable defaults on observational data:

```rust
use semicontinuous_models::{FitOptions, LogNormalOptions};

let two_part_options = FitOptions::stable_defaults();
let lognormal_options = LogNormalOptions::stable_defaults();
```

## Public entry points

- `fit_two_part_input`
- `fit_two_part_weighted_input`
- `fit_two_part_clustered_input`
- `fit_tweedie_input`
- `fit_lognormal_smearing_input`
- `select_models_input`
- `cross_validate_models_input`
- `compare_models_input`

Preprocess helpers:

- `drop_constant_columns`
- `column_has_variation`
- `outcome_diagnostics`

## Data contract

- Provide an intercept column explicitly if you want one.
- Outcomes are expected to be non-negative and shaped as `n x 1`.
- Weights (if provided) must be strictly positive and shaped as `n x 1`.
- Cluster labels (if provided) must have length `n`.

## Examples

From crate root:

```bash
cargo run --example two_part
cargo run --example tweedie
cargo run --example selection
cargo run --example compare_models
```
