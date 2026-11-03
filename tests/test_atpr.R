# Optional integration test; requires the external metrics-adjuster CLI.
# All observations below are artificial and created in memory.
source("metrics/load.R")
ensure_packages(c("data.table", "pROC"))
command <- build_metrics_adjuster_command()
ensure_metrics_adjuster_available(command)
set.seed(71)
group <- rep(0:1, each = 300L)
risk <- plogis(rnorm(length(group)))
outcome <- rbinom(length(group), 1, risk)
for (reference in 0:1) {
  values <- compute_atpr_values_for_attribute(outcome, risk, group, reference,
    "artificial_integration", command)
  ordinary_tpr <- mean(risk[group == reference & outcome == 1] >= classification_threshold)
  stopifnot(identical(names(values), c("0", "1")), all(is.finite(values)),
            abs(values[[as.character(reference)]] - ordinary_tpr) < 1e-10)
}
cat("PASS: external aTPR integration for both reference directions;\n",
    "reference subgroup retains ordinary TPR.\n")
