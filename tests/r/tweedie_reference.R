# Reference fits for tests/tweedie_reference.rs.
# Run: Rscript tests/r/tweedie_reference.R tests/reference/tweedie_statmod.json
# Needs statmod and jsonlite.
suppressMessages({ library(statmod); library(jsonlite) })

out <- commandArgs(trailingOnly = TRUE)[1]

n <- 60
x1 <- ((seq_len(n) * 37) %% 23) / 23 - 0.5
x2 <- as.numeric(seq_len(n) %% 3 == 0)
mu_true <- exp(0.4 + 0.8 * x1 - 0.5 * x2)
# Deterministic response with exact zeros: zero where a pseudo-uniform falls below
# exp(-mu), otherwise a scaled positive value with skew.
u <- ((seq_len(n) * 7919) %% 101) / 101
v <- ((seq_len(n) * 104729) %% 97 + 1) / 98
y <- ifelse(u < exp(-1.2 * mu_true), 0, mu_true * (-log(v)) * 1.5)
stopifnot(sum(y == 0) >= 8, sum(y > 0) >= 20)
w <- 0.5 + ((seq_len(n) * 13) %% 7) / 3
X <- cbind(1, x1, x2)

fit_case <- function(p, weights) {
  fit <- glm(y ~ x1 + x2, weights = weights,
             family = statmod::tweedie(var.power = p, link.power = 0),
             control = glm.control(epsilon = 1e-14, maxit = 200))
  beta <- unname(coef(fit))
  mu <- unname(fitted(fit))
  # HC0 sandwich on the quasi-score, the form the crate uses:
  # bread X'diag(w mu^(2-p))X, score w (y - mu) mu^(1-p) x.
  bread <- solve(crossprod(X, X * (weights * mu^(2 - p))))
  score <- X * (weights * (y - mu) * mu^(1 - p))
  cov <- bread %*% crossprod(score) %*% bread
  # Model-based covariance with dispersion fixed at 1, for reference.
  list(power = p, weighted = !all(weights == 1),
       beta = beta, se_robust = unname(sqrt(diag(cov))),
       se_model_unit_dispersion = unname(sqrt(diag(vcov(fit) / summary(fit)$dispersion))),
       converged = fit$converged)
}

cases <- list()
for (p in c(1.2, 1.5, 1.8)) {
  cases[[length(cases) + 1]] <- fit_case(p, rep(1, n))
  cases[[length(cases) + 1]] <- fit_case(p, w)
}
write_json(list(x1 = x1, x2 = x2, y = y, weights = w, cases = cases,
                statmod = as.character(packageVersion("statmod")),
                R = R.version.string),
           out, digits = NA, auto_unbox = TRUE, pretty = TRUE)
