#!/usr/bin/env Rscript
# EZ hierarchical reference fit for VPW08 (Chavez & Vandekerckhove 2025 Appendix E).

suppressPackageStartupMessages({
  library(R2jags)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else normalizePath(getwd())
out_dir <- file.path(root, "results", "vpw08")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

data_path <- file.path(root, "data", "vpw08", "vpw08.csv")
model_path <- file.path(out_dir, "ez_appendix_e_model.bug")
json_path <- file.path(out_dir, "ez_fit.json")

set.seed(15)

data_raw <- read.csv(data_path)
colnames(data_raw) <- c("sub", "change_quality", "change_type", "noChange", "response", "rt")

tmp <- data_raw[data_raw$rt <= 3, ]
change <- 1 - tmp$noChange
cond <- rep(0, nrow(tmp))
cond[tmp$change_quality == 0 & tmp$change_type == 0] <- 1
cond[tmp$change_quality == 1 & tmp$change_type == 0] <- 2
cond[tmp$change_quality == 0 & tmp$change_type == 1] <- 3
cond[tmp$change_quality == 1 & tmp$change_type == 1] <- 4
cond[change == 0] <- 5

data <- data.frame(
  sub = tmp$sub,
  cond = cond,
  change = change,
  change_quality = tmp$change_quality,
  change_type = tmp$change_type,
  response = tmp$response,
  rt = tmp$rt
)

ez_summaries <- function(data) {
  output <- c()
  for (i in sort(unique(data$sub))) {
    for (k in sort(unique(data$cond))) {
      subset <- data[data$sub == i & data$cond == k, ]
      output <- rbind(
        output,
        c(
          sub = unique(subset$sub),
          cond = unique(subset$cond),
          change = unique(subset$change),
          change_quality = unique(subset$change_quality),
          change_type = unique(subset$change_type),
          nTrials = nrow(subset),
          score = sum(subset$response),
          meanRT = mean(subset$rt),
          varRT = var(subset$rt)
        )
      )
    }
  }
  as.data.frame(output)
}

ezdata <- ez_summaries(data)

write(
  "
model {
  drift_mu ~ dnorm(0, 1)
  drift_lambda ~ dgamma(2, 1)
  drift_sigma = pow(drift_lambda, -0.5)
  for (i in 1:4) {
    gamma[i] ~ dnorm(0, 1)
  }
  for (j in 1:5) {
    drift_pred[j] <- drift_mu + A[j] * (gamma[1] * B[j] + gamma[2] * C[j] + gamma[3] * B[j] * C[j]) + (1 - A[j]) * gamma[4]
  }
  for (k in 1:length(nTrials)) {
    bound[k] ~ dgamma(2, 1)
    nondt[k] ~ dexp(1)
    drift[k] ~ dnorm(drift_pred[cond[k]], drift_lambda)
    ey[k] <- exp(-bound[k] * drift[k])
    Pc[k] <- 1 / (1 + ey[k])
    PRT[k] <- 2 * pow(drift[k], 3) / bound[k] * pow(ey[k] + 1, 2) / (2 * -bound[k] * drift[k] * ey[k] - ey[k] * ey[k] + 1)
    MDT[k] <- (bound[k] / (2 * drift[k])) * (1 - ey[k]) / (1 + ey[k])
    MRT[k] <- MDT[k] + nondt[k]
    correct[k] ~ dbin(Pc[k], nTrials[k])
    varRT[k] ~ dnorm(1 / PRT[k], 0.5 * (nTrials[k] - 1) * PRT[k] * PRT[k])
    meanRT[k] ~ dnorm(MRT[k], PRT[k] * nTrials[k])
  }
}
",
  model_path
)

data_toJAGS <- list(
  nTrials = ezdata$nTrials,
  meanRT = ezdata$meanRT,
  varRT = ezdata$varRT,
  correct = ezdata$score,
  cond = ezdata$cond,
  A = ezdata$change,
  B = ezdata$change_quality,
  C = ezdata$change_type
)

parameters <- c("gamma", "drift_mu", "drift_lambda", "drift_pred", "drift", "bound", "nondt")
n.chains <- 4
myinits <- vector("list", n.chains)
for (i in seq_len(n.chains)) {
  myinits[[i]] <- list(drift = rnorm(nrow(ezdata), 0, 1))
}

cat("[vpw08:ez] Running JAGS (2500 iter, 250 burnin, 4 chains)...\n")
samples <- jags(
  data = data_toJAGS,
  parameters.to.save = parameters,
  model.file = model_path,
  n.chains = n.chains,
  n.iter = 2500,
  n.burnin = 250,
  n.thin = 1,
  DIC = TRUE,
  inits = myinits
)

epsilon <- 0.1
prior_constant <- pnorm(epsilon) - pnorm(-epsilon)
gamma_sims <- samples$BUGSoutput$sims.list$gamma
bf <- c()
for (i in 1:3) {
  post_mass <- mean(gamma_sims[, i] > -epsilon & gamma_sims[, i] < epsilon)
  bf <- c(bf, prior_constant / post_mass)
}

summ <- samples$BUGSoutput$summary
pick <- function(name) {
  row <- as.list(summ[name, c("mean", "sd", "2.5%", "97.5%", "Rhat")])
  list(
    mean = unname(row$mean),
    sd = unname(row$sd),
    lo95 = unname(row[["2.5%"]]),
    hi95 = unname(row[["97.5%"]]),
    rhat = unname(row$Rhat)
  )
}

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org")
}
library(jsonlite)

out <- list(
  model = "ez_appendix_e",
  source = "Chavez & Vandekerckhove (2025) Appendix E",
  mcmc = list(n_iter = 2500, n_burnin = 250, n_chains = 4, seed = 15),
  drift_mu = pick("drift_mu"),
  drift_lambda = pick("drift_lambda"),
  gamma = list(
    gamma_1 = pick("gamma[1]"),
    gamma_2 = pick("gamma[2]"),
    gamma_3 = pick("gamma[3]"),
    gamma_4 = pick("gamma[4]")
  ),
  drift_pred = list(
    `1` = pick("drift_pred[1]"),
    `2` = pick("drift_pred[2]"),
    `3` = pick("drift_pred[3]"),
    `4` = pick("drift_pred[4]"),
    `5` = pick("drift_pred[5]")
  ),
  nondt_pooled_mean = mean(samples$BUGSoutput$sims.list$nondt),
  bound_pooled_mean = mean(samples$BUGSoutput$sims.list$bound),
  bayes_factors_gamma1_3 = list(
    gamma_1 = bf[[1]],
    gamma_2 = bf[[2]],
    gamma_3 = bf[[3]]
  ),
  convergence = list(
    max_rhat = max(summ[, "Rhat"], na.rm = TRUE),
    converged = max(summ[, "Rhat"], na.rm = TRUE) < 1.05,
    rhat_gate = 1.05
  )
)
write_json(out, json_path, pretty = TRUE, auto_unbox = TRUE)
cat("[vpw08:ez] Wrote", json_path, "\n")
