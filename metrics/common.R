sensitive_vars <- c("age80", "male", "white", "private")

ensure_packages <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop("Missing required R package(s): ", paste(missing, collapse = ", "))
  }
}

validate_ehr_data <- function(dat, name = "data", sensitive_columns = sensitive_vars) {
  if (!is.data.frame(dat) || !nrow(dat)) {
    stop(name, " must be a non-empty data.frame.")
  }
  # data.table and tibble inherit data.frame but have different column-selection
  # semantics; normalize once and leave the caller's object unchanged.
  dat <- as.data.frame(dat)
  dat <- dat[, setdiff(names(dat), c("subject_id", "hadm_id")), drop = FALSE]
  if (anyDuplicated(names(dat))) {
    stop(name, " contains duplicate column names.")
  }
  if (!all(sensitive_columns %in% names(dat))) {
    stop(name, " is missing sensitive columns: ",
         paste(setdiff(sensitive_columns, names(dat)), collapse = ", "))
  }
  for (column in names(dat)) {
    input <- dat[[column]]
    if (is.factor(input)) input <- as.character(input)
    values <- suppressWarnings(as.numeric(input))
    if (anyNA(values) || any(!values %in% c(0, 1))) {
      stop(name, "$", column, " must contain complete binary 0/1 values.")
    }
    dat[[column]] <- as.integer(values)
  }
  dat
}

check_same_columns <- function(reference, candidate, candidate_name = "synthetic_data") {
  if (!setequal(names(reference), names(candidate))) {
    stop(candidate_name, " must have the same columns as the real data.")
  }
  candidate[, names(reference), drop = FALSE]
}

as_replicate_list <- function(x, name = "synthetic_data") {
  if (is.data.frame(x)) {
    return(list(repetition_1 = x))
  }
  if (!is.list(x) || !length(x) || !all(vapply(x, is.data.frame, logical(1)))) {
    stop(name, " must be a data.frame or a non-empty list of data.frames.")
  }
  if (is.null(names(x)) || anyNA(names(x)) || any(!nzchar(names(x))) || anyDuplicated(names(x))) {
    stop(name, " must use unique, non-empty repetition names.")
  }
  x
}

get_phenotype_vars <- function(dat, sensitive_columns = sensitive_vars) {
  setdiff(names(dat), sensitive_columns)
}

as_binary_matrix <- function(dat, columns) {
  mat <- as.matrix(dat[, columns, drop = FALSE])
  storage.mode(mat) <- "numeric"
  mat
}

make_sensitive_subgroup_grid <- function(sensitive_columns = sensitive_vars) {
  do.call(rbind, lapply(sensitive_columns, function(attribute) {
    data.frame(
      sensitive_attribute = attribute,
      group_value = c(0L, 1L),
      stringsAsFactors = FALSE
    )
  }))
}

safe_mean <- function(x) {
  if (!length(x) || all(is.na(x))) {
    return(NA_real_)
  }
  mean(x, na.rm = TRUE)
}

standard_error <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) <= 1) {
    return(NA_real_)
  }
  stats::sd(x) / sqrt(length(x))
}

compute_eval_phenotypes <- function(
  real_train,
  prevalence_threshold = 0.01,
  sensitive_columns = sensitive_vars,
  evaluation_phenotypes = NULL
) {
  ensure_packages("data.table")
  real_train <- validate_ehr_data(real_train, "real_train", sensitive_columns)
  if (length(prevalence_threshold) != 1L || !is.finite(prevalence_threshold) ||
      prevalence_threshold < 0 || prevalence_threshold >= 1) {
    stop("prevalence_threshold must be a single number in [0, 1).")
  }
  phenotype_vars <- get_phenotype_vars(real_train, sensitive_columns)
  if (!length(phenotype_vars)) {
    stop("At least one phenotype column is required.")
  }
  if (!is.null(evaluation_phenotypes) &&
      (!is.character(evaluation_phenotypes) || !length(evaluation_phenotypes) ||
       anyNA(evaluation_phenotypes) || anyDuplicated(evaluation_phenotypes) ||
       any(!evaluation_phenotypes %in% phenotype_vars))) {
    stop("evaluation_phenotypes must be unique names of existing phenotype columns.")
  }
  subgroup_grid <- make_sensitive_subgroup_grid(sensitive_columns)
  phenotype_matrix <- as_binary_matrix(real_train, phenotype_vars)

  prevalence_rows <- vector("list", nrow(subgroup_grid))
  for (i in seq_len(nrow(subgroup_grid))) {
    attribute <- subgroup_grid$sensitive_attribute[i]
    group_value <- subgroup_grid$group_value[i]
    in_group <- real_train[[attribute]] == group_value
    n_in_group <- sum(in_group)
    if (n_in_group == 0) {
      n_positive <- rep(NA_integer_, length(phenotype_vars))
      prevalence <- rep(NA_real_, length(phenotype_vars))
    } else {
      subgroup <- phenotype_matrix[in_group, , drop = FALSE]
      n_positive <- colSums(subgroup == 1)
      prevalence <- colMeans(subgroup == 1)
    }
    prevalence_rows[[i]] <- data.frame(
      sensitive_attribute = attribute,
      group_value = group_value,
      n_in_subgroup = n_in_group,
      phenotype = phenotype_vars,
      n_positive = as.integer(n_positive),
      prevalence = as.numeric(prevalence),
      stringsAsFactors = FALSE
    )
  }

  prevalence_table <- data.table::rbindlist(prevalence_rows)
  by_phenotype <- split(prevalence_table, prevalence_table$phenotype)
  phenotype_summary <- data.table::rbindlist(lapply(names(by_phenotype), function(phenotype) {
    values <- by_phenotype[[phenotype]]
    data.frame(
      phenotype = phenotype,
      minimum_subgroup_prevalence = if (anyNA(values$prevalence)) {
        NA_real_
      } else {
        min(values$prevalence)
      },
      prevalence_threshold = prevalence_threshold,
      stringsAsFactors = FALSE
    )
  }))
  phenotype_summary[, passes_prevalence_screen := !is.na(minimum_subgroup_prevalence) &
    minimum_subgroup_prevalence > prevalence_threshold]
  phenotype_summary[, included := passes_prevalence_screen]
  phenotype_summary[, selection_rule := "real_training_prevalence"]
  if (!is.null(evaluation_phenotypes)) {
    # Sensitivity analyses retain the main benchmark's selected features, even
    # when an alternative subgroup definition changes their subgroup prevalence.
    phenotype_summary[, included := phenotype %in% evaluation_phenotypes]
    phenotype_summary[, selection_rule := "supplied_fixed_phenotypes"]
  }
  data.table::setorder(phenotype_summary, -included, -minimum_subgroup_prevalence, phenotype)

  list(
    phenotypes = if (is.null(evaluation_phenotypes)) {
      phenotype_summary[included == TRUE, phenotype]
    } else evaluation_phenotypes,
    prevalence_by_subgroup = prevalence_table,
    phenotype_screen = phenotype_summary
  )
}

summarize_repetitions <- function(metrics, metric_columns, group_columns) {
  metrics <- data.table::as.data.table(metrics)
  missing_columns <- setdiff(c(metric_columns, group_columns), names(metrics))
  if (length(missing_columns)) {
    stop("Missing summary columns: ", paste(missing_columns, collapse = ", "))
  }
  metrics[, c(
    setNames(lapply(.SD, safe_mean), metric_columns),
    setNames(lapply(.SD, standard_error), paste0(metric_columns, "_se")),
    setNames(lapply(.SD, function(x) sum(is.finite(x))), paste0(metric_columns, "_n_valid")),
    list(n_repetitions = .N)
  ), by = group_columns, .SDcols = metric_columns]
}
