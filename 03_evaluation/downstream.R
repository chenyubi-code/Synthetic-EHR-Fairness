evaluate_prediction_subgroups <- function(
  method,
  repetition,
  regime,
  model,
  outcome,
  evaluation_data,
  probability,
  sensitive_columns,
  command_spec,
  temporary_directory
) {
  rows <- list()
  observed <- evaluation_data[[outcome]]

  for (attribute in sensitive_columns) {
    adjusted_tpr <- lapply(atpr_reference_groups, function(reference_group) {
      compute_atpr_values_for_attribute(
        observed,
        probability,
        evaluation_data[[attribute]],
        reference_group,
        paste(method, repetition, regime, model, outcome, attribute,
              paste0("ref", reference_group), sep = "__"),
        command_spec,
        temporary_directory
      )
    })

    for (group_value in c(0L, 1L)) {
      in_group <- evaluation_data[[attribute]] == group_value
      values <- compute_binary_metrics(
        observed[in_group], probability[in_group], classification_threshold
      )
      names(values)[names(values) == "AUROC"] <- "AUC"
      rows[[length(rows) + 1L]] <- data.frame(
        method = method,
        repetition = repetition,
        regime = regime,
        prediction_model = model,
        phenotype = outcome,
        sensitive_attribute = attribute,
        group_value = group_value,
        n_test = sum(in_group),
        as.list(values),
        aTPR_ref0 = adjusted_tpr[[1]][[as.character(group_value)]],
        aTPR_ref1 = adjusted_tpr[[2]][[as.character(group_value)]],
        stringsAsFactors = FALSE
      )
    }
  }
  data.table::rbindlist(rows)
}

compute_dsge <- function(subgroup_metrics) {
  metric_names <- c("AUC", "ACC", "TPR", "FPR", "PPR", "aTPR_ref0", "aTPR_ref1")
  long <- data.table::melt(
    data.table::as.data.table(subgroup_metrics),
    id.vars = c("method", "repetition", "regime", "prediction_model", "phenotype",
                "sensitive_attribute", "group_value"),
    measure.vars = metric_names,
    variable.name = "metric",
    value.name = "metric_value"
  )
  gaps <- data.table::dcast(
    long,
    method + repetition + regime + prediction_model + phenotype + sensitive_attribute + metric ~ group_value,
    value.var = "metric_value"
  )
  data.table::setnames(gaps, c("0", "1"), c("metric_group0", "metric_group1"))
  gaps[, subgroup_gap := metric_group1 - metric_group0]

  trtr_reference <- gaps[regime == "TRTR", .(
    prediction_model,
    phenotype,
    sensitive_attribute,
    metric,
    trtr_subgroup_gap = subgroup_gap
  )]
  differences <- merge(
    gaps[regime != "TRTR"],
    trtr_reference,
    by = c("prediction_model", "phenotype", "sensitive_attribute", "metric"),
    all.x = TRUE,
    sort = FALSE
  )
  differences[, DSGE := abs(subgroup_gap - trtr_subgroup_gap)]
  differences[, signed_gap_change := subgroup_gap - trtr_subgroup_gap]
  differences[, metric := as.character(metric)]
  differences[, atpr_reference_group := as.integer(ifelse(
    grepl("^aTPR_ref", metric), sub("^aTPR_ref", "", metric), NA_character_
  ))]

  atpr <- differences[grepl("^aTPR_ref", metric)]
  atpr[, metric := "aTPR"]
  atpr <- atpr[, {
    scores <- DSGE
    scores[is.na(scores)] <- -Inf
    selected <- if (all(is.infinite(scores) & scores < 0)) 1L else which.max(scores)
    .SD[selected]
  }, by = c("method", "repetition", "regime", "prediction_model", "phenotype",
            "sensitive_attribute", "metric")]

  data.table::rbindlist(list(
    differences[!grepl("^aTPR_ref", metric)],
    atpr
  ), use.names = TRUE)
}

compute_downstream_metrics <- function(
  real_train,
  real_test,
  synthetic_train = NULL,
  synthetic_test = NULL,
  method = "synthetic",
  regimes = c("TSTR", "TSRTR", "TRTS"),
  outcomes = downstream_outcomes,
  prevalence_threshold = 0.01,
  minimum_test_cases = minimum_test_cases_per_subgroup,
  sensitive_columns = sensitive_vars,
  temporary_directory = tempdir()
) {
  ensure_packages(c("data.table", "glmnet", "pROC", "Matrix", "randomForest", "xgboost"))
  regimes <- unique(regimes)
  if (!length(regimes) || any(!regimes %in% c("TSTR", "TSRTR", "TRTS"))) {
    stop("regimes must be a subset of TSTR, TSRTR, and TRTS.")
  }
  needs_train <- any(c("TSTR", "TSRTR") %in% regimes)
  needs_test <- "TRTS" %in% regimes
  if (needs_train && is.null(synthetic_train)) {
    stop("synthetic_train is required for TSTR and TSRTR.")
  }
  if (needs_test && is.null(synthetic_test)) {
    stop("synthetic_test is required for TRTS.")
  }

  train_replicates <- if (is.null(synthetic_train)) NULL else {
    as_replicate_list(synthetic_train, "synthetic_train")
  }
  test_replicates <- if (is.null(synthetic_test)) NULL else {
    as_replicate_list(synthetic_test, "synthetic_test")
  }
  repetition_names <- if (needs_train) names(train_replicates) else names(test_replicates)
  if (needs_train && needs_test && !setequal(names(train_replicates), names(test_replicates))) {
    stop("synthetic_train and synthetic_test must use the same repetition names.")
  }

  real_train <- validate_ehr_data(real_train, "real_train", sensitive_columns)
  real_test <- validate_ehr_data(real_test, "real_test", sensitive_columns)
  real_test <- check_same_columns(real_train, real_test, "real_test")
  screen <- compute_eval_phenotypes(real_train, prevalence_threshold, sensitive_columns)
  evaluation_phenotypes <- screen$phenotypes
  eligible_outcomes <- get_prediction_phenotypes(
    real_train,
    real_test,
    prevalence_threshold,
    minimum_test_cases,
    sensitive_columns
  )
  missing_outcomes <- setdiff(outcomes, eligible_outcomes)
  if (length(missing_outcomes)) {
    stop("Outcomes do not meet the prevalence and test-case criteria: ",
         paste(missing_outcomes, collapse = ", "))
  }

  command_spec <- build_metrics_adjuster_command()
  ensure_metrics_adjuster_available(command_spec)
  rows <- list()
  fixed_models <- list()

  for (outcome in outcomes) {
    predictors <- setdiff(evaluation_phenotypes, outcome)
    x_real_train <- as_binary_matrix(real_train, predictors)
    x_real_test <- as_binary_matrix(real_test, predictors)
    fixed_models[[outcome]] <- list()

    for (i in seq_along(prediction_models)) {
      model <- prediction_models[i]
      set.seed(analysis_seed + i)
      fitted <- fit_fixed_real_model(model, x_real_train, real_train[[outcome]])
      fixed_models[[outcome]][[model]] <- fitted
      probability <- predict_fixed_real_model(fitted, x_real_test)
      rows[[length(rows) + 1L]] <- evaluate_prediction_subgroups(
        "real", "reference", "TRTR", model, outcome, real_test, probability,
        sensitive_columns, command_spec, temporary_directory
      )
    }
  }

  for (repetition in repetition_names) {
    synthetic_train_data <- NULL
    synthetic_test_data <- NULL
    if (needs_train) {
      synthetic_train_data <- validate_ehr_data(
        train_replicates[[repetition]], paste0("synthetic_train$", repetition), sensitive_columns
      )
      synthetic_train_data <- check_same_columns(real_train, synthetic_train_data, repetition)
    }
    if (needs_test) {
      synthetic_test_data <- validate_ehr_data(
        test_replicates[[repetition]], paste0("synthetic_test$", repetition), sensitive_columns
      )
      synthetic_test_data <- check_same_columns(real_train, synthetic_test_data, repetition)
      if (nrow(synthetic_test_data) != nrow(real_test)) {
        stop("Each synthetic test replicate must have the same number of rows as real_test.")
      }
    }

    for (outcome in outcomes) {
      predictors <- setdiff(evaluation_phenotypes, outcome)
      x_real_train <- as_binary_matrix(real_train, predictors)
      x_real_test <- as_binary_matrix(real_test, predictors)
      if (needs_train) {
        x_synthetic_train <- as_binary_matrix(synthetic_train_data, predictors)
      }

      for (i in seq_along(prediction_models)) {
        model <- prediction_models[i]
        for (regime in regimes) {
          evaluation_data <- real_test
          if (regime == "TRTS") {
            evaluation_data <- synthetic_test_data
            probability <- predict_fixed_real_model(
              fixed_models[[outcome]][[model]],
              as_binary_matrix(synthetic_test_data, predictors)
            )
          } else {
            set.seed(analysis_seed + if (regime == "TSRTR") 100L + i else i)
            training_matrix <- if (regime == "TSRTR") {
              rbind(x_real_train, x_synthetic_train)
            } else {
              x_synthetic_train
            }
            training_outcome <- if (regime == "TSRTR") {
              c(real_train[[outcome]], synthetic_train_data[[outcome]])
            } else {
              synthetic_train_data[[outcome]]
            }
            fitted <- fit_fixed_real_model(model, training_matrix, training_outcome)
            probability <- predict_fixed_real_model(fitted, x_real_test)
          }
          rows[[length(rows) + 1L]] <- evaluate_prediction_subgroups(
            method, repetition, regime, model, outcome, evaluation_data, probability,
            sensitive_columns, command_spec, temporary_directory
          )
        }
      }
    }
  }

  subgroup_metrics <- data.table::rbindlist(rows)
  dsge_by_repetition <- compute_dsge(subgroup_metrics)
  repetition_means <- dsge_by_repetition[, .(mean_DSGE = safe_mean(DSGE)),
    by = c("method", "repetition", "regime", "prediction_model", "metric")]
  summary <- repetition_means[, .(
    mean_DSGE = safe_mean(mean_DSGE),
    se = standard_error(mean_DSGE)
  ), by = c("method", "regime", "prediction_model", "metric")]
  predictors <- data.table::rbindlist(lapply(outcomes, function(outcome) {
    data.frame(outcome = outcome, predictor = setdiff(evaluation_phenotypes, outcome))
  }))

  list(
    subgroup_metrics = subgroup_metrics,
    dsge_by_repetition = dsge_by_repetition,
    summary = summary,
    phenotype_screen = screen$phenotype_screen,
    predictors = predictors
  )
}
