compute_privacy_population <- function(
  real_data,
  synthetic_data,
  method = "synthetic",
  sensitive_columns = sensitive_vars,
  population = "mem"
) {
  ensure_packages(c("data.table", "Rcpp"))
  real_data <- validate_ehr_data(real_data, "real_data", sensitive_columns)
  replicates <- as_replicate_list(synthetic_data)
  phenotypes <- get_phenotype_vars(real_data, sensitive_columns)
  if (!length(phenotypes)) {
    stop("At least one phenotype column is required.")
  }

  if (length(sensitive_columns) < 2L) stop("AIR requires at least two sensitive attributes.")
  rows <- list()
  target_rows <- list()
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
        accuracy_values <- f1_values

        for (target_attribute in target_attributes) {
          ones <- tied_ones[, target_attribute]
          zeros <- tie_count - ones
          prediction <- ifelse(ones > zeros, 1L, 0L)
          prediction[tie_count == 0L] <- NA_integer_
          vote_ties <- tie_count > 0L & ones == zeros
          if (any(vote_ties)) {
            set.seed(stable_air_vote_seed(
              method, if (population == "mem") repetition else paste0(repetition, "|nonmem"),
              known_attribute, group_value, target_attribute
            ), kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
            prediction[vote_ties] <- sample(c(0L, 1L), sum(vote_ties), replace = TRUE)
          }
          f1_values[[target_attribute]] <- f1_score_binary(
            real_data[[target_attribute]][real_index], prediction
          )
          accuracy_values[[target_attribute]] <- safe_mean(
            real_data[[target_attribute]][real_index] == prediction)
          target_rows[[length(target_rows) + 1L]] <- data.frame(
            method = method, repetition = repetition, population = population,
            sensitive_attribute = known_attribute, group_value = group_value,
            target_attribute = target_attribute, F1 = f1_values[[target_attribute]],
            ACC = accuracy_values[[target_attribute]],
            n_random_vote_ties = sum(vote_ties), stringsAsFactors = FALSE)
        }

        rows[[length(rows) + 1L]] <- data.frame(
          method = method,
          repetition = repetition,
          population = population,
          sensitive_attribute = known_attribute,
          group_value = group_value,
          n_real_records = length(real_index),
          n_synthetic_candidates = length(synthetic_index),
          MIR = if (length(real_index) && length(synthetic_index)) safe_mean(nearest$distance) else NA_real_,
          AIR = safe_mean(f1_values),
          AIR_ACC = safe_mean(accuracy_values),
          n_valid_AIR_targets = sum(is.finite(f1_values)),
          stringsAsFactors = FALSE
        )
      }
    }
  }

  by_repetition <- data.table::rbindlist(rows)
  by_repetition[, MIR_normalized := MIR / length(phenotypes)]
  list(by_repetition = by_repetition, target_metrics = data.table::rbindlist(target_rows))
}

# Contrast population-level scores for the same synthetic replicate. This pairs
# subgroup cells, never individual training and test patients.
adjust_privacy_exposure <- function(population_metrics) {
  ensure_packages("data.table")
  raw <- data.table::as.data.table(population_metrics)
  keys <- c("method", "repetition", "sensitive_attribute", "group_value")
  values <- c("MIR", "MIR_normalized", "AIR", "AIR_ACC", "n_real_records",
              "n_synthetic_candidates", "n_valid_AIR_targets")
  required <- c(keys, values, "population")
  if (!all(required %in% names(raw))) stop("Incomplete population metric columns.")
  if (anyDuplicated(raw[, c(keys, "population"), with = FALSE])) stop("Duplicate privacy population cells.")
  if (any(!raw$population %in% c("mem", "nonmem"))) stop("population must be mem or nonmem.")
  member <- raw[population == "mem", c(keys, values), with = FALSE]
  nonmember <- raw[population == "nonmem", c(keys, values), with = FALSE]
  adjusted <- merge(member, nonmember, by = keys, suffixes = c("_mem", "_nonmem"),
                    all = TRUE, sort = FALSE)
  if (!nrow(adjusted) || nrow(adjusted) != nrow(member) || nrow(adjusted) != nrow(nonmember) ||
      anyNA(adjusted$n_real_records_mem) || anyNA(adjusted$n_real_records_nonmem)) {
    stop("Member/nonmember subgroup cells must be complete and paired.")
  }
  if (any(adjusted$n_synthetic_candidates_mem != adjusted$n_synthetic_candidates_nonmem)) {
    stop("Member and nonmember scores must use the same synthetic candidate pools.")
  }
  adjusted[, `:=`(baMIR = MIR_nonmem - MIR_mem,
                  baMIR_normalized = MIR_normalized_nonmem - MIR_normalized_mem,
                  baAIR = AIR_mem - AIR_nonmem,
                  baAIR_ACC = AIR_ACC_mem - AIR_ACC_nonmem)]
  adjusted[, valid := is.finite(baMIR) & is.finite(baAIR)]
  adjusted
}

compute_privacy_metrics <- function(
  real_train,
  real_test,
  synthetic_data,
  method = "synthetic",
  sensitive_columns = sensitive_vars
) {
  ensure_packages(c("data.table", "Rcpp"))
  real_train <- validate_ehr_data(real_train, "real_train", sensitive_columns)
  real_test <- validate_ehr_data(real_test, "real_test", sensitive_columns)
  real_test <- check_same_columns(real_train, real_test, "real_test")
  members <- compute_privacy_population(real_train, synthetic_data, method, sensitive_columns, "mem")
  nonmembers <- compute_privacy_population(real_test, synthetic_data, method, sensitive_columns, "nonmem")
  raw <- data.table::rbindlist(list(members$by_repetition, nonmembers$by_repetition))
  adjusted <- adjust_privacy_exposure(raw)
  gaps <- data.table::dcast(
    adjusted,
    method + repetition + sensitive_attribute ~ group_value,
    value.var = c("baMIR", "baAIR", "MIR_mem", "MIR_nonmem", "AIR_mem", "AIR_nonmem")
  )
  gaps[, baMIRG := abs(baMIR_1 - baMIR_0)]
  gaps[, baAIRG := abs(baAIR_1 - baAIR_0)]
  gaps[, valid := is.finite(baMIRG) & is.finite(baAIRG)]
  summary <- summarize_repetitions(
    gaps,
    metric_columns = c("baMIRG", "baAIRG"),
    group_columns = c("method", "sensitive_attribute")
  )
  list(population_metrics = raw, by_repetition = adjusted, gaps = gaps, summary = summary,
       subgroup_summary = summarize_repetitions(adjusted,
         c("baMIR", "baAIR", "MIR_mem", "MIR_nonmem", "AIR_mem", "AIR_nonmem"),
         c("method", "sensitive_attribute", "group_value")),
       target_metrics = data.table::rbindlist(list(members$target_metrics, nonmembers$target_metrics)))
}
