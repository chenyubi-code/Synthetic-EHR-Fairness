# Size-matched candidate pools from Supplementary Methods S1.5.
# This source requires metrics/load.R (or common.R, privacy.R and nearest_neighbor.R).
# Only aggregate metrics are returned; no query-level predictions or matches.
.matched_cpp_path <- file.path(dirname(normalizePath(sys.frame(1)$ofile,
                                                    mustWork = TRUE)), "batch_neighbors.cpp")
.batch_privacy_kernel <- local({
  cpp_path <- .matched_cpp_path
  compiled <- new.env(parent = globalenv())
  function() {
    if (!exists("batch_nearest_hamming", envir = compiled, inherits = FALSE)) {
      Rcpp::sourceCpp(cpp_path, env = compiled, showOutput = FALSE)
    }
    compiled$batch_nearest_hamming
  }
})
rm(.matched_cpp_path)

# Use all targets, never silently average a smaller set if a target F1 is undefined.
.matched_population_scores <- function(nearest, truth, sensitive_columns, method,
                                       repetition, attribute, group, population,
                                       pool_sizes, n_phenotypes) {
  targets <- setdiff(sensitive_columns, attribute)
  rows <- list()
  for (pool in seq_along(pool_sizes)) {
    count <- nearest$tie_count[, pool]
    f1 <- acc <- numeric(length(targets))
    for (j in seq_along(targets)) {
      target <- targets[j]
      ones <- nearest$tied_ones[[match(target, sensitive_columns)]][, pool]
      prediction <- as.integer(ones > count - ones)
      ties <- ones == count - ones
      if (any(ties)) {
        label <- if (population == "mem") repetition else paste0(repetition, "|nonmem")
        set.seed(stable_air_vote_seed(method, label, attribute, group, target),
                 kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
        prediction[ties] <- sample(c(0L, 1L), sum(ties), replace = TRUE)
      }
      f1[j] <- f1_score_binary(truth[[target]], prediction)
      acc[j] <- mean(truth[[target]] == prediction)
    }
    if (any(!is.finite(f1))) {
      stop("Undefined target F1 in matched-pool evaluation: ", repetition, "/",
           population, "/", attribute, "=", group,
           ". All AIR targets are required; no incomplete averages are returned.")
    }
    mir <- mean(nearest$distance[, pool])
    rows[[pool]] <- data.frame(method = method, repetition = repetition,
      sensitive_attribute = attribute, group_value = group, population = population,
      resample = pool - 1L, n_real_records = nrow(truth),
      n_synthetic_candidates = pool_sizes[pool], MIR = mir,
      MIR_normalized = mir / n_phenotypes,
      AIR = mean(f1), AIR_ACC = mean(acc),
      n_valid_AIR_targets = sum(is.finite(f1)), stringsAsFactors = FALSE)
  }
  data.table::rbindlist(rows)
}

compute_matched_privacy_metrics <- function(
  real_train,
  real_test,
  synthetic_data,
  method = "synthetic",
  sensitive_columns = sensitive_vars,
  n_resamples = 20L
) {
  ensure_packages(c("data.table", "Rcpp"))
  if (!is.numeric(n_resamples) || length(n_resamples) != 1L || !is.finite(n_resamples) ||
      n_resamples < 1L || n_resamples > 31L || n_resamples != as.integer(n_resamples)) {
    stop("n_resamples must be an integer from 1 to 31; use 20 for the manuscript.")
  }
  if (length(sensitive_columns) < 2L || anyDuplicated(sensitive_columns)) {
    stop("Specify at least two unique sensitive attributes for AIR.")
  }
  real_train <- validate_ehr_data(real_train, "real_train", sensitive_columns)
  real_test <- validate_ehr_data(real_test, "real_test", sensitive_columns)
  real_test <- check_same_columns(real_train, real_test, "real_test")
  replicates <- as_replicate_list(synthetic_data)
  phenotypes <- get_phenotype_vars(real_train, sensitive_columns)
  if (!length(phenotypes)) stop("At least one phenotype column is required.")
  kernel <- .batch_privacy_kernel()
  rows <- list()
  designs <- list()
  for (repetition in names(replicates)) {
    synthetic <- validate_ehr_data(replicates[[repetition]], repetition, sensitive_columns)
    synthetic <- check_same_columns(real_train, synthetic, repetition)
    for (attribute in sensitive_columns) {
      counts <- vapply(0:1, function(g) sum(synthetic[[attribute]] == g), integer(1))
      matched_size <- min(counts)
      if (matched_size == 0L) stop("Matched pools require both synthetic subgroups: ", attribute)
      for (group in 0:1) {
        candidates <- which(synthetic[[attribute]] == group)
        pools <- c(list(candidates), lapply(seq_len(n_resamples), function(b) {
          set.seed(stable_air_vote_seed("candidate_pool_size_v1", method, repetition,
                                        attribute, b, group),
                   kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
          sort(candidates[sample.int(length(candidates), matched_size, replace = FALSE)])
        }))
        membership <- matrix(0L, length(candidates), n_resamples + 1L)
        for (j in seq_along(pools)) membership[match(pools[[j]], candidates), j] <- 1L
        pool_sizes <- lengths(pools)
        designs[[length(designs) + 1L]] <- data.frame(method = method,
          repetition = repetition, sensitive_attribute = attribute, group_value = group,
          full_pool_size = length(candidates), matched_pool_size = matched_size,
          n_resamples = n_resamples, stringsAsFactors = FALSE)
        for (population in c("mem", "nonmem")) {
          real <- if (population == "mem") real_train else real_test
          query <- real[real[[attribute]] == group, , drop = FALSE]
          if (!nrow(query)) stop("Matched pools require both real subgroups: ", population, "/", attribute)
          labels <- as.matrix(synthetic[candidates, sensitive_columns, drop = FALSE])
          storage.mode(labels) <- "integer"
          nearest <- kernel(as_binary_matrix(query, phenotypes),
                            as_binary_matrix(synthetic[candidates, , drop = FALSE], phenotypes),
                            labels, membership)
          rows[[length(rows) + 1L]] <- .matched_population_scores(nearest, query,
            sensitive_columns, method, repetition, attribute, group, population,
            pool_sizes, length(phenotypes))
        }
      }
    }
  }
  raw <- data.table::rbindlist(rows)
  adjusted <- data.table::rbindlist(lapply(0:n_resamples, function(b) {
    result <- adjust_privacy_exposure(raw[resample == b])
    result[, resample := b]
    result
  }))
  gaps <- data.table::dcast(adjusted,
    method + repetition + sensitive_attribute + resample ~ group_value,
    value.var = c("baMIR", "baAIR"))
  gaps[, `:=`(baMIRG = abs(baMIR_1 - baMIR_0), baAIRG = abs(baAIR_1 - baAIR_0))]
  # A missing draw propagates to its repetition average; incomplete draws must not
  # silently receive a different denominator.
  average_draws <- function(x, keys, metrics) {
    full <- data.table::copy(x[resample == 0L, c(keys, metrics), with = FALSE])
    full[, condition := "full"]
    matched <- x[resample > 0L, lapply(.SD, mean), by = keys, .SDcols = metrics]
    matched[, condition := "matched"]
    data.table::rbindlist(list(full, matched), use.names = TRUE)
  }
  exposure_metrics <- c("baMIR", "baAIR", "MIR_mem", "MIR_nonmem", "AIR_mem", "AIR_nonmem")
  gap_metrics <- c("baMIRG", "baAIRG")
  subgroup_keys <- c("method", "repetition", "sensitive_attribute", "group_value")
  gap_keys <- c("method", "repetition", "sensitive_attribute")
  exposure_repetitions <- average_draws(adjusted, subgroup_keys, exposure_metrics)
  gap_repetitions <- average_draws(gaps, gap_keys, gap_metrics)
  list(population_metrics = raw, pool_design = data.table::rbindlist(designs),
       by_resample = adjusted, gaps_by_resample = gaps,
       by_repetition = exposure_repetitions, gaps_by_repetition = gap_repetitions,
       subgroup_summary = summarize_repetitions(exposure_repetitions, exposure_metrics,
         c("method", "sensitive_attribute", "group_value", "condition")),
       summary = summarize_repetitions(gap_repetitions, gap_metrics,
         c("method", "sensitive_attribute", "condition")))
}
