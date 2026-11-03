# Artificial examples only. Run from the repository root: Rscript tests/test_downstream.R
source("metrics/load.R")
library(data.table)

make_row <- function(regime, group, value, a0 = value, a1 = value) {
  data.frame(method = if (regime == "TRTR") "real" else "toy",
             repetition = if (regime == "TRTR") "reference" else "r1",
             regime = regime, prediction_model = "ridge_logistic", phenotype = "outcome",
             sensitive_attribute = "age80", group_value = group,
             AUC = value, ACC = value, TPR = value, FPR = value, PPR = value,
             aTPR_ref0 = a0, aTPR_ref1 = a1)
}
scores <- rbind(make_row("TRTR", 0, .6, .6, .7), make_row("TRTR", 1, .8, .8, .9),
                make_row("TSTR", 0, .5, .5, .4), make_row("TSTR", 1, .7, .9, .8))
gaps <- compute_dsge(scores)
absolute <- compute_dsae(scores)
# The same downward shift in both groups preserves the ordinary gap but not performance.
stopifnot(all(abs(gaps[metric != "aTPR", DSGE]) < 1e-12),
          all(abs(absolute$by_repetition[metric != "aTPR", DSAE] - .1) < 1e-12))
# aTPR selects the larger reference-specific *error* independently for each subgroup.
stopifnot(abs(absolute$by_repetition[metric == "aTPR" & group_value == 0, DSAE] - .3) < 1e-12,
          abs(absolute$by_repetition[metric == "aTPR" & group_value == 1, DSAE] - .1) < 1e-12,
          abs(gaps[metric == "aTPR", DSGE] - .2) < 1e-12)
scores$aTPR_ref1[3] <- NA_real_
stopifnot(is.na(compute_dsge(scores)[metric == "aTPR", DSGE]),
          is.na(compute_dsae(scores)$by_repetition[metric == "aTPR" & group_value == 0, DSAE]))
stopifnot(inherits(try(compute_dsge(scores[-1, ]), silent = TRUE), "try-error"))

set.seed(81)
toy <- as.data.frame(lapply(seq_len(8), function(i) rbinom(512, 1, .5)))
names(toy) <- c(sensitive_vars, "CV_404", "GU_582", "P1", "P2")
real_train <- toy[1:256, ]
real_test <- toy[257:512, ]
# Three classifiers, all three synthetic regimes, and a fixed-model proxy-test identity.
result <- compute_downstream_metrics(real_train, real_test,
  synthetic_train = list(r1 = real_train), synthetic_test = list(r1 = real_test),
  method = "identity", outcomes = "CV_404", minimum_test_cases = 10,
  include_atpr = FALSE)
stopifnot(setequal(result$subgroup_metrics$prediction_model, prediction_models),
          setequal(result$dsge_by_repetition$regime, c("TSTR", "TSRTR", "TRTS")),
          all(result$dsge_by_repetition[regime == "TRTS", DSGE] == 0),
          all(result$dsae_by_repetition[regime == "TRTS", DSAE] == 0),
          all(is.finite(result$subgroup_metrics$AUC)),
          !"aTPR" %in% result$dsge_by_repetition$metric,
          all(!result$predictors$predictor %in% c(sensitive_vars, "CV_404")))
cat("PASS: downstream formulas, aTPR reference reduction, missing-value handling,\n",
    "three classifiers, three regimes, and fixed TRTR/TRTS model identity.\n")
