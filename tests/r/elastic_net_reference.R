# Reference fits for the elastic_net_wls test in src/models/two_part/elastic_net.rs.
# Run: Rscript tests/r/elastic_net_reference.R tests/reference/elastic_net_glmnet.json
# Needs glmnet and jsonlite.
#
# The crate minimises 0.5 * sum w_i (z_i - x_i b)^2
#   + lambda * (alpha * |b|_1 + (1 - alpha) / 2 * |b|_2^2)
# over the columns of x as given, with no standardisation and the first column
# (the intercept) optionally unpenalised. glmnet scales the weights to sum to one, so
# its lambda is lambda / sum(w). glmnet also standardises the response by its weighted
# standard deviation s and reports lambda on the original scale, which leaves the L1
# penalty alone and divides the ridge penalty by s. Its objective therefore equals the
# crate's only if (lambda_g, alpha_g) is chosen so that lambda_g * alpha_g =
# lambda / sum(w) * alpha and lambda_g * (1 - alpha_g) = s * lambda / sum(w) * (1 - alpha).
# An unpenalised intercept is glmnet's own intercept on
# the remaining columns. glmnet skips constant columns, so a penalised intercept
# column cannot be fitted with it, and those cases carry no reference coefficients.
suppressMessages({ library(glmnet); library(jsonlite) })

out <- commandArgs(trailingOnly = TRUE)[1]

n <- 80
k <- 5
x <- cbind(1, sapply(seq_len(k - 1), function(j) {
  sin(seq_len(n) * (0.7 + 0.31 * j)) * (1 + 0.4 * j) + 0.3 * j
}))
beta_true <- c(0.5, 1.2, 0, -0.8, 0.05)
z <- as.vector(x %*% beta_true) + 0.6 * cos(seq_len(n) * 1.9)
w <- 0.4 + ((seq_len(n) * 11) %% 9) / 4

fit_case <- function(alpha, lambda, exclude_intercept) {
  beta <- numeric(0)
  if (exclude_intercept) {
    wn <- w / sum(w)
    s <- sqrt(sum(wn * (z - sum(wn * z))^2))
    scaled <- lambda / sum(w)
    lambda_g <- scaled * (alpha + s * (1 - alpha))
    alpha_g <- alpha / (alpha + s * (1 - alpha))
    fit <- glmnet(x[, -1], z, family = "gaussian", weights = w, alpha = alpha_g,
                  lambda = lambda_g, intercept = TRUE, standardize = FALSE,
                  control = list(thresh = 1e-20, maxit = 1e7))
    beta <- as.vector(coef(fit))
  }
  list(alpha = alpha, lambda = lambda, exclude_intercept = exclude_intercept,
       beta = beta)
}

cases <- list()
for (exclude in c(TRUE, FALSE)) {
  for (alpha in c(0.2, 0.5, 1.0)) {
    for (lambda in c(0.5, 5, 30)) {
      cases[[length(cases) + 1]] <- fit_case(alpha, lambda, exclude)
    }
  }
}
write_json(list(x = x, z = z, weights = w, cases = cases,
                glmnet = as.character(packageVersion("glmnet")),
                R = R.version.string),
           out, digits = NA, auto_unbox = TRUE)
