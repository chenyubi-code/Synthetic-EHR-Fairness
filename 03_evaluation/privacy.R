compute_privacy_metrics <- function(
  real_data,
  synthetic_data,
  method = "synthetic",
  sensitive_columns = sensitive_vars
) {
  ensure_packages(c("data.table", "Rcpp"))
  real_data <- validate_ehr_data(real_data, "real_data", sensitive_columns)
  replicates <- as_replicate_list(synthetic_data)
  phenotypes <- get_phenotype_vars(real_data, sensitive_columns)
  if (!length(phenotypes)) {
    stop("At least one phenotype column is required.")
  }

  rows <- list()
  for (repetition in names(replicates)) {
    synthetic <- validate_ehr_data(replicates[[repetition]], repetition, sensitive_columns)
    synthetic <- check_same_columns(real_data, synthetic, repetition)

    for (known_attribute in sensitive_columns) {
      for (group_value in c(0L, 1L)) {
        real_index <- which(real_data[[known_attribute]] == group_value)
        synthetic_index <- which(synthetic[[known_attribute]] == group_value)
        nearest <- nearest_hamming_tie_vote(
          as_binary_matrix(real_data[real_index, , drop = FALSE], phenotypes),
          as_binary_matrix(synthetic[synthetic_index, , drop = FALSE], phenotypes),
          as.matrix(synthetic[synthetic_index, sensitive_columns, drop = FALSE])
        )

        tied_ones <- as.matrix(nearest$tied_ones)
        colnames(tied_ones) <- sensitive_columns
        tie_count <- as.integer(nearest$tie_count)
        target_attributes <- setdiff(sensitive_columns, known_attribute)
        f1_values <- setNames(numeric(length(target_attributes)), target_attributes)

        for (target_attribute in target_attributes) {
          ones <- tied_ones[, target_attribute]
          zeros <- tie_count - ones
          prediction <- ifelse(ones > zeros, 1L, 0L)
          prediction[tie_count == 0L] <- NA_integer_
          vote_ties <- tie_count > 0L & ones == zeros
          if (any(vote_ties)) {
            set.seed(stable_air_vote_seed(
              method, repetition, known_attribute, group_value, target_attribute
            ))
            prediction[vote_ties] <- sample(c(0L, 1L), sum(vote_ties), replace = TRUE)
          }
          f1_values[[target_attribute]] <- f1_score_binary(
            real_data[[target_attribute]][real_index], prediction
          )
        }

        rows[[length(rows) + 1L]] <- data.frame(
          method = method,
          repetition = repetition,
          sensitive_attribute = known_attribute,
          group_value = group_value,
          n_real_records = length(real_index),
          n_synthetic_candidates = length(synthetic_index),
          MIR = mean(nearest$distance, na.rm = TRUE),
          AIR = safe_mean(f1_values),
          stringsAsFactors = FALSE
        )
      }
    }
  }

  by_repetition <- data.table::rbindlist(rows)
  gaps <- data.table::dcast(
    by_repetition,
    method + repetition + sensitive_attribute ~ group_value,
    value.var = c("MIR", "AIR")
  )
  gaps[, MIRG := abs(MIR_1 - MIR_0)]
  gaps[, AIRG := abs(AIR_1 - AIR_0)]
  summary <- summarize_repetitions(
    gaps,
    metric_columns = c("MIR_0", "MIR_1", "AIR_0", "AIR_1", "MIRG", "AIRG"),
    group_columns = c("method", "sensitive_attribute")
  )

  list(by_repetition = by_repetition, gaps = gaps, summary = summary)
}
