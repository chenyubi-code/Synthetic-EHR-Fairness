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

# Two generation means (.15 and .35) give mean .25 and SE .10. Task/model
# observations are not independent generation replicates for uncertainty.
aggregation_cells <- CJ(repetition = c("r1", "r2"),
                        phenotype = c("task1", "task2"),
                        prediction_model = c("model1", "model2"))
aggregation_cells[, `:=`(method = "toy", regime = "TSTR", metric = "AUC",
                         sensitive_attribute = "age80", group_value = 0L,
                         DSAE = c(0, .1, .2, .3, .2, .3, .4, .5))]
aggregated <- summarize_downstream_errors(aggregation_cells, "DSAE",
                                          c("sensitive_attribute", "group_value"))
stopifnot(abs(aggregated$summary$mean_DSAE - .25) < 1e-12,
          abs(aggregated$summary$mean_DSAE_se - .1) < 1e-12,
          aggregated$summary$n_valid == 8L,
          aggregated$summary$n_repetitions == 2L)
aggregation_cells$DSAE[1] <- NA_real_
incomplete <- summarize_downstream_errors(aggregation_cells, "DSAE")
stopifnot(is.na(incomplete$summary$mean_DSAE),
          incomplete$summary$n_valid == 7L,
          incomplete$summary$n_valid_repetitions == 1L)

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
stopifnot(all(result$dsge_summary_by_attribute[regime == "TRTS", mean_DSGE] == 0),
          all(result$dsae_summary_by_subgroup[regime == "TRTS", mean_DSAE] == 0),
          all(result$dsge_summary$n_expected == 12L),
          all(result$dsae_summary$n_expected == 24L))
# Selecting a single classifier must retain its original benchmark random seed.
# Explicit phenotypes also bypass a changed prevalence screen without changing
# outcome test-case checks or accidentally including sensitive attributes.
fixed_phenotypes <- compute_eval_phenotypes(real_train)$phenotypes
selected_model <- compute_downstream_metrics(real_train, real_test,
  synthetic_test = real_test, regimes = "TRTS", models = "random_forest",
  outcomes = "CV_404", minimum_test_cases = 10, include_atpr = FALSE,
  prevalence_threshold = .6, evaluation_phenotypes = fixed_phenotypes)
full_reference <- result$subgroup_metrics[regime == "TRTR" & prediction_model == "random_forest"]
selected_reference <- selected_model$subgroup_metrics[regime == "TRTR"]
stopifnot(identical(full_reference$AUC, selected_reference$AUC),
          setequal(selected_model$predictors$predictor, setdiff(fixed_phenotypes, "CV_404")),
          all(selected_model$phenotype_screen$included),
          all(!selected_model$phenotype_screen$passes_prevalence_screen))
cat("PASS: downstream formulas, aTPR reference reduction, missing-value handling,\n",
    "repetition-level aggregation, fixed phenotype selection, classifier seeds,\n",
    "three classifiers, three regimes, and fixed TRTR/TRTS model identity.\n")
