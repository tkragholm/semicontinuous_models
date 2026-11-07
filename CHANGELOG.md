# Changelog

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
