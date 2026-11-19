# Changelog

## Unreleased

Breaking: statistics that do not exist are no longer reported as numbers.

- `autocorrelation` returns `Option<f64>`, `None` for a constant chain or a lag not
  shorter than the chain, where it returned 0.
- `effective_sample_size` returns `Option<f64>`, `None` for a constant chain or fewer
  than two draws.
- `MtpConvergenceSummary::alpha_split_rhat` and `beta_split_rhat` hold `Option<f64>`,
  `None` where a half-chain is constant. A constant chain, as a sampler that accepts no
  proposal produces, used to report a split R-hat of 1. `kappa_split_rhat` is `None`
  likewise, and `max_split_rhat` is `None` when any tracked parameter is undefined.
  `MtpConvergenceSummary::is_converged(threshold)` is false unless every split R-hat
  is defined and below the threshold.
- `summarize_multi_chain_convergence_tracking` takes whether `kappa` is a sampled
  parameter. The multi-chain fit leaves a fixed `kappa` (log-normal positive part) out
  of the diagnostics.
- `utils::calculate_quantile` returns `Option<f64>`, `None` for no values. The
  posterior and effect summaries of an empty sample report NaN instead of 0.

Deterministic fits are bit-identical to 0.6.0.

Tests: the Tweedie fitter is checked against `glm` with `statmod::tweedie` at powers
1.2, 1.5 and 1.8, and the elastic-net coordinate descent against `glmnet`; the R
scripts and reference output are in `tests/r` and `tests/reference`. The MTP recovery
test compares the truth with the posterior mean in posterior standard deviations.

## 0.6.0 - 2026-11-19

All random draws now come from `epistat_rng::Stream` (SplitMix64) instead of
`rand`'s `StdRng`: the case bootstrap, the cross-validation fold shuffle and the
MTP samplers. `rand_distr` still supplies the gamma, normal, chi-square and
Wishart draws, from the same stream. A seed gives the same results on every
platform, but they are not the same as in 0.5.0, so bootstrap intervals,
cross-validation fold assignments and MTP posterior draws change for a given
seed. Fits without random draws are bit-identical to 0.5.0.

- Chain `i` of a multi-chain MTP fit is seeded with
  `epistat_rng::replicate_seed(seed, i)`. `MtpMultiChainOptions::seed_stride` and
  `MtpError::InvalidSeedStride` are removed, and chain 0 no longer uses the base
  seed itself.
- The prepared designs factorise through `epistat-linalg`'s
  `Factor::cholesky_factor` and `Factor::solve`, with the same pivot checks.
- Builds on `epistat-linalg` 0.2.

## 0.5.0

The dense solves, the cached factor of the gamma and Tweedie IRLS, the
row-blocked `X'WX`, `X'Wz` and `X beta` products, the sandwich covariance, the
type 7 quantile, the normal quantile and the MCMC diagnostics now come from the
epistat crates (`epistat-linalg`, `epistat-inference`, `epistat-stats`).

- Coefficients and predictions are bit-identical to 0.4.1 on the golden
  fixtures. Robust standard errors can differ in the last bit, because the
  cluster sums no longer use fused multiply-add.
- Factorisations always run on one thread and the row blocks keep their fixed
  order, so results do not depend on the number of threads.
  `FitOptions::inner_parallel` and `TweedieOptions::inner_parallel` are kept
  for compatibility and no longer change anything.
- The sandwich covariance factorises the information matrix with the first
  method that succeeds, then solves with it.

## 0.4.1

Additions only. The results of the existing functions are unchanged: fits with the
Tweedie and two-part options compared bit for bit with 0.4.0 on aarch64 macOS.

- `models::prepared` fits one design repeatedly under different weights, as a
  fractional-random-weight bootstrap does. `PreparedGammaDesign` and
  `PreparedTwoPartDesign` read the design once into a compact form, and `PreparedPredictionDesign` averages fitted means
  over rows. Their results agree with the dense fits up to rounding and are not
  bit-identical to them. A constructor returns `None` for a design the compact form
  cannot represent, and a fit returns a `PreparedFitError` when it cannot give a
  reliable answer, so a caller holds the dense fit as a fallback. See the module
  documentation for the rules.
- `fit_tweedie_ref_warm_prevalidated_with_scratch` and `TweedieScratch` let a caller
  that runs many fits on one thread keep one centred design and one set of working
  columns rather than allocating them per fit.
- Tweedie fits take the saturation check from the linear predictor the line search
  holds, and the Gram and cross-product loops run through slices.
