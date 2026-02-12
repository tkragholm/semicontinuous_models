# Examples

These examples illustrate the primary modeling workflows using `ModelInput`.

## API overview

Primary entry points (use `ModelInput`):

- `fit_two_part_input`
- `fit_two_part_weighted_input`
- `fit_two_part_clustered_input`
- `fit_tweedie_input`
- `fit_lognormal_smearing_input`
- `select_models_input`

## Two-part model

```
cargo run --example two_part
```

Builds a two-part model (logit + gamma log-link), runs a small bootstrap, and
prints the first expected outcome.

## Tweedie GLM

```
cargo run --example tweedie
```

Fits a Tweedie GLM (p = 1.5) and prints the first predicted mean.

## Selection workflow

```
cargo run --example selection
```

Runs the Park test and compares candidate Tweedie powers against a log-normal
model, reporting the candidate count, Park slope, and model metrics
(RMSE/MAE/RMSLE/R2/deviance).

The library also exposes `cross_validate_models_input` for K-fold evaluation
with the same metrics.

## Compare models

```
cargo run --example compare_models
```

Fits two-part (default + elastic net), Tweedie, and log-normal models and
prints RMSE/MAE/RMSLE/R2 tables for in-sample and cross-validated metrics,
plus a Park test summary with candidate metrics.
