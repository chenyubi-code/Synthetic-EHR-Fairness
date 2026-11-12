subgroup_prevalence <- function(dat, phenotypes, sensitive_attribute, group_value) {
  in_group <- dat[[sensitive_attribute]] == group_value
  if (!any(in_group)) {
    return(rep(NA_real_, length(phenotypes)))
  }
  colMeans(as_binary_matrix(dat[in_group, , drop = FALSE], phenotypes) == 1)
}

subgroup_correlations <- function(dat, phenotypes, sensitive_attribute, group_value) {
  in_group <- dat[[sensitive_attribute]] == group_value
  n_pairs <- choose(length(phenotypes), 2)
  if (sum(in_group) < 3) {
    return(rep(NA_real_, n_pairs))
  }
  correlation_matrix <- suppressWarnings(stats::cor(
    as_binary_matrix(dat[in_group, , drop = FALSE], phenotypes)
  ))
  as.numeric(correlation_matrix[upper.tri(correlation_matrix)])
}

compute_data_level_metrics <- function(
  real_data,
  synthetic_data,
  method = "synthetic",
  prevalence_threshold = 0.01,
  sensitive_columns = sensitive_vars,
  evaluation_phenotypes = NULL
) {
  ensure_packages("data.table")
  real_data <- validate_ehr_data(real_data, "real_data", sensitive_columns)
  replicates <- as_replicate_list(synthetic_data)
  screen <- compute_eval_phenotypes(real_data, prevalence_threshold, sensitive_columns,
                                   evaluation_phenotypes)
  phenotypes <- screen$phenotypes
  if (length(phenotypes) < 2) {
    stop("At least two evaluation phenotypes are required.")
  }

  real_reference <- setNames(lapply(sensitive_columns, function(attribute) {
    prevalence <- lapply(0:1, function(group) subgroup_prevalence(real_data, phenotypes, attribute, group))
    correlations <- lapply(0:1, function(group) subgroup_correlations(real_data, phenotypes, attribute, group))
    list(
      prevalence = prevalence,
      correlations = correlations,
      sensitive_prevalence = mean(real_data[[attribute]] == 1),
      phenotype_prevalence_gap =
        subgroup_prevalence(real_data, phenotypes, attribute, 1L) -
        subgroup_prevalence(real_data, phenotypes, attribute, 0L),
      phenotype_correlation_gap =
        subgroup_correlations(real_data, phenotypes, attribute, 1L) -
        subgroup_correlations(real_data, phenotypes, attribute, 0L)
    )
  }), sensitive_columns)

  rows <- list()
  subgroup_rows <- list()
  for (repetition in names(replicates)) {
    synthetic <- validate_ehr_data(replicates[[repetition]], repetition, sensitive_columns)
    synthetic <- check_same_columns(real_data, synthetic, repetition)
    for (attribute in sensitive_columns) {
      reference <- real_reference[[attribute]]
      synthetic_prevalence <- lapply(0:1, function(group) subgroup_prevalence(synthetic, phenotypes, attribute, group))
      synthetic_correlations <- lapply(0:1, function(group) subgroup_correlations(synthetic, phenotypes, attribute, group))
      prevalence_gap <-
        subgroup_prevalence(synthetic, phenotypes, attribute, 1L) -
        subgroup_prevalence(synthetic, phenotypes, attribute, 0L)
      correlation_gap <-
        subgroup_correlations(synthetic, phenotypes, attribute, 1L) -
        subgroup_correlations(synthetic, phenotypes, attribute, 0L)
      correlation_error <- abs(correlation_gap - reference$phenotype_correlation_gap)

      rows[[length(rows) + 1L]] <- data.frame(
        method = method,
        repetition = repetition,
        sensitive_attribute = attribute,
        SARE = abs(mean(synthetic[[attribute]] == 1) - reference$sensitive_prevalence),
        MPPGE = safe_mean(abs(prevalence_gap - reference$phenotype_prevalence_gap)),
        MPCGE = safe_mean(correlation_error),
        n_eval_phenotypes = length(phenotypes),
        n_valid_prevalence_gaps = sum(is.finite(prevalence_gap - reference$phenotype_prevalence_gap)),
        n_correlation_pairs = length(correlation_error),
        n_valid_correlation_pairs = sum(!is.na(correlation_error)),
        stringsAsFactors = FALSE
      )
      for (group in 0:1) {
        prevalence_error <- abs(synthetic_prevalence[[group + 1L]] - reference$prevalence[[group + 1L]])
        correlation_error <- abs(synthetic_correlations[[group + 1L]] - reference$correlations[[group + 1L]])
        subgroup_rows[[length(subgroup_rows) + 1L]] <- data.frame(
          method = method, repetition = repetition, sensitive_attribute = attribute,
          group_value = group,
          n_real_records = sum(real_data[[attribute]] == group),
          n_synthetic_records = sum(synthetic[[attribute]] == group),
          MAPPE = safe_mean(prevalence_error), MAPCE = safe_mean(correlation_error),
          n_eval_phenotypes = length(phenotypes),
          n_valid_prevalences = sum(is.finite(prevalence_error)),
          n_correlation_pairs = length(correlation_error),
          n_valid_correlation_pairs = sum(is.finite(correlation_error)),
          stringsAsFactors = FALSE
        )
      }
    }
  }

  by_repetition <- data.table::rbindlist(rows)
  subgroup_by_repetition <- data.table::rbindlist(subgroup_rows)
  summary <- summarize_repetitions(
    by_repetition,
    metric_columns = c("SARE", "MPPGE", "MPCGE"),
    group_columns = c("method", "sensitive_attribute")
  )

  list(
    by_repetition = by_repetition,
    subgroup_by_repetition = subgroup_by_repetition,
    summary = summary,
    subgroup_summary = summarize_repetitions(subgroup_by_repetition,
      c("MAPPE", "MAPCE"), c("method", "sensitive_attribute", "group_value")),
    phenotype_screen = screen$phenotype_screen
  )
}
