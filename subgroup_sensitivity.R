# Alternative subgroup definitions in Supplementary Methods S1.6.
# Input and returned frames stay in memory; no records are written here.
# Apply to cross-sectional and HALO frames before fitting fresh generators.
# Do not relabel previously generated synthetic records.
recode_sensitive_definition <- function(data, raw_attributes,
                                        scenario = c("age85", "black")) {
  scenario <- match.arg(scenario)
  data <- as.data.frame(data)
  raw_attributes <- as.data.frame(raw_attributes)
  if (anyDuplicated(names(data)) || anyDuplicated(names(raw_attributes))) {
    stop("Input column names must be unique.")
  }
  old <- if (scenario == "age85") "age80" else "white"
  raw_column <- if (scenario == "age85") "age" else "ethnicity"
  if (!all(c("subject_id", old) %in% names(data)) ||
      !all(c("subject_id", raw_column) %in% names(raw_attributes))) {
    stop("data needs subject_id and ", old, "; raw_attributes needs subject_id and ", raw_column, ".")
  }
  if (scenario %in% names(data)) stop("The new sensitive column already exists: ", scenario)
  if (anyNA(data$subject_id) || anyNA(raw_attributes$subject_id) ||
      anyDuplicated(raw_attributes$subject_id)) {
    stop("IDs must be complete and raw_attributes must contain one row per patient.")
  }
  index <- match(data$subject_id, raw_attributes$subject_id)
  if (anyNA(index)) stop("Some input patients are absent from raw_attributes.")
  if (scenario == "age85") {
    age <- suppressWarnings(as.numeric(as.character(raw_attributes$age[index])))
    if (any(!is.finite(age))) stop("Baseline ages must be complete finite numbers.")
    original <- as.integer(age >= 80)
    replacement <- as.integer(age >= 85)
  } else {
    ethnicity <- toupper(trimws(as.character(raw_attributes$ethnicity[index])))
    original <- as.integer(!is.na(ethnicity) & ethnicity == "WHITE")
    replacement <- as.integer(!is.na(ethnicity) & startsWith(ethnicity, "BLACK"))
  }
  if (anyNA(data[[old]]) || !all(data[[old]] == original)) {
    stop("Original ", old, " disagrees with the patient-level source attributes.")
  }
  output <- data
  names(output)[names(output) == old] <- scenario
  output[[scenario]] <- replacement
  unchanged <- setdiff(names(data), old)
  stopifnot(identical(data[, unchanged, drop = FALSE], output[, unchanged, drop = FALSE]))
  output
}

prepare_subgroup_sensitivity <- function(real_train, real_test, raw_attributes,
                                         evaluation_phenotypes,
                                         scenario = c("age85", "black")) {
  scenario <- match.arg(scenario)
  if (!is.data.frame(real_train) || !is.data.frame(real_test) ||
      !"subject_id" %in% names(real_train) || !"subject_id" %in% names(real_test)) {
    stop("Real train/test inputs must retain subject_id for the fixed patient split.")
  }
  if (!nrow(real_train) || !nrow(real_test) || anyDuplicated(real_train$subject_id) ||
      anyDuplicated(real_test$subject_id) ||
      length(intersect(real_train$subject_id, real_test$subject_id))) {
    stop("Train and test must be nonempty, patient-level, and disjoint.")
  }
  if (!identical(names(real_train), names(real_test))) stop("Train/test schemas must match.")
  sensitive <- c("age80", "male", "white", "private")
  if (!is.character(evaluation_phenotypes) || !length(evaluation_phenotypes) ||
      anyNA(evaluation_phenotypes) || anyDuplicated(evaluation_phenotypes) ||
      !all(evaluation_phenotypes %in% setdiff(names(real_train), c("subject_id", "hadm_id", sensitive)))) {
    stop("Supply the frozen phenotype names from the primary benchmark, before recoding.")
  }
  sensitive[sensitive == if (scenario == "age85") "age80" else "white"] <- scenario
  list(real_train = recode_sensitive_definition(real_train, raw_attributes, scenario),
       real_test = recode_sensitive_definition(real_test, raw_attributes, scenario),
       sensitive_columns = sensitive, evaluation_phenotypes = evaluation_phenotypes,
       scenario = scenario)
}
