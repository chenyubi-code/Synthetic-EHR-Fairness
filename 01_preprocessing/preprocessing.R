# Core preprocessing for the cross-sectional benchmark and HALO inputs.

.lowercase_names <- function(data) {
  data <- as.data.frame(data)
  names(data) <- tolower(names(data))
  if (anyDuplicated(names(data))) {
    stop("Column names must be unique after conversion to lower case.")
  }
  data
}

.require_columns <- function(data, required, table_name) {
  missing <- setdiff(required, names(data))
  if (length(missing) > 0L) {
    stop(table_name, " is missing: ", paste(missing, collapse = ", "))
  }
}

.prepare_admissions <- function(admissions, patients) {
  admissions <- .lowercase_names(admissions)
  patients <- .lowercase_names(patients)

  .require_columns(
    admissions,
    c("subject_id", "hadm_id", "insurance", "ethnicity", "admittime"),
    "admissions"
  )
  .require_columns(patients, c("subject_id", "gender", "dob"), "patients")

  admissions <- admissions |>
    dplyr::select(subject_id, hadm_id, insurance, ethnicity, admittime) |>
    dplyr::left_join(
      dplyr::select(patients, subject_id, gender, dob),
      by = "subject_id"
    )

  inconsistent_ids <- admissions |>
    dplyr::group_by(subject_id) |>
    dplyr::summarise(
      n_insurance = dplyr::n_distinct(insurance, na.rm = TRUE),
      n_ethnicity = dplyr::n_distinct(ethnicity, na.rm = TRUE),
      n_gender = dplyr::n_distinct(gender, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::filter(n_insurance > 1L | n_ethnicity > 1L | n_gender > 1L) |>
    dplyr::pull(subject_id)

  admissions <- admissions |>
    dplyr::filter(!subject_id %in% inconsistent_ids) |>
    dplyr::mutate(
      admittime = as.POSIXct(admittime),
      dob = as.POSIXct(dob)
    )

  baseline_age <- admissions |>
    dplyr::group_by(subject_id) |>
    dplyr::slice_min(admittime, with_ties = FALSE) |>
    dplyr::ungroup() |>
    dplyr::transmute(
      subject_id,
      age = as.numeric(difftime(admittime, dob, units = "days")) / 365.242,
      age = ifelse(age > 89, 90, age)
    )

  admissions |>
    dplyr::select(-admittime, -dob) |>
    dplyr::left_join(baseline_age, by = "subject_id") |>
    dplyr::select(subject_id, hadm_id, age, insurance, ethnicity, gender)
}

.map_diagnoses <- function(diagnoses, phecode_map) {
  diagnoses <- .lowercase_names(diagnoses)
  phecode_map <- .lowercase_names(phecode_map)

  .require_columns(
    diagnoses,
    c("subject_id", "hadm_id", "icd9_code"),
    "diagnoses"
  )
  .require_columns(
    phecode_map,
    c("vocabulary_id", "code", "phecode"),
    "phecode_map"
  )

  mapping <- phecode_map |>
    dplyr::transmute(
      vocabulary_id = as.character(vocabulary_id),
      code = as.character(code),
      phecode = as.character(phecode)
    ) |>
    dplyr::filter(vocabulary_id == "ICD9CM") |>
    dplyr::distinct(code, .keep_all = TRUE)

  diagnoses |>
    dplyr::transmute(
      subject_id,
      hadm_id,
      icd9_code = as.character(icd9_code)
    ) |>
    dplyr::filter(!is.na(icd9_code), icd9_code != "") |>
    dplyr::distinct(subject_id, hadm_id, icd9_code) |>
    dplyr::mutate(code = as.character(icd::short_to_decimal(icd9_code))) |>
    dplyr::inner_join(mapping, by = "code") |>
    dplyr::transmute(
      subject_id,
      hadm_id,
      phenotype = stringr::str_extract(phecode, "^[A-Z]+_[0-9]+")
    ) |>
    dplyr::filter(!is.na(phenotype)) |>
    dplyr::distinct()
}

.prepare_visits <- function(admissions, patients, diagnoses, phecode_map) {
  visits <- .prepare_admissions(admissions, patients)
  mapped <- .map_diagnoses(diagnoses, phecode_map)
  if (nrow(mapped) == 0L) return(visits)

  phenotype_matrix <- mapped |>
    dplyr::mutate(value = 1L) |>
    tidyr::pivot_wider(
      names_from = phenotype,
      values_from = value,
      values_fill = 0L
    )
  phenotype_columns <- sort(setdiff(
    names(phenotype_matrix), c("subject_id", "hadm_id")
  ))

  visits |>
    dplyr::left_join(phenotype_matrix, by = c("subject_id", "hadm_id")) |>
    dplyr::mutate(
      dplyr::across(
        dplyr::all_of(phenotype_columns),
        ~ as.integer(tidyr::replace_na(.x, 0L))
      )
    ) |>
    dplyr::select(
      subject_id, hadm_id, age, insurance, ethnicity, gender,
      dplyr::all_of(phenotype_columns)
    )
}

.encode_sensitive_attributes <- function(data, id_columns, phenotype_columns) {
  data |>
    dplyr::mutate(
      age80 = as.integer(age >= 80),
      male = as.integer(gender == "M"),
      white = as.integer(ethnicity == "WHITE"),
      private = as.integer(insurance == "Private")
    ) |>
    dplyr::select(
      dplyr::all_of(id_columns), age80, male, white, private,
      dplyr::all_of(phenotype_columns)
    ) |>
    dplyr::mutate(dplyr::across(dplyr::everything(), as.integer))
}

prepare_cross_sectional_data <- function(
    admissions,
    patients,
    diagnoses,
    phecode_map,
    min_phenotype_count = 20L,
    train_fraction = 0.8,
    split_seed = 123L) {
  visits <- .prepare_visits(admissions, patients, diagnoses, phecode_map)
  phenotype_columns <- setdiff(
    names(visits),
    c("subject_id", "hadm_id", "age", "insurance", "ethnicity", "gender")
  )

  patients_wide <- visits |>
    dplyr::group_by(subject_id) |>
    dplyr::summarise(
      age = dplyr::first(age),
      insurance = dplyr::first(insurance),
      ethnicity = dplyr::first(ethnicity),
      gender = dplyr::first(gender),
      dplyr::across(
        dplyr::all_of(phenotype_columns),
        ~ max(.x, na.rm = TRUE)
      ),
      .groups = "drop"
    ) |>
    dplyr::filter(age >= 18)

  if (length(phenotype_columns) > 0L) {
    prevalence <- colSums(patients_wide[phenotype_columns], na.rm = TRUE)
    phenotype_columns <- sort(names(prevalence)[
      prevalence >= min_phenotype_count
    ])
  }

  full <- .encode_sensitive_attributes(
    patients_wide,
    id_columns = "subject_id",
    phenotype_columns = phenotype_columns
  )
  strata <- with(full, paste(age80, male, white, private, sep = "_"))
  set.seed(split_seed)
  train_index <- as.vector(caret::createDataPartition(
    strata,
    p = train_fraction,
    list = FALSE
  ))

  list(
    full = full,
    train = full[train_index, , drop = FALSE],
    test = full[-train_index, , drop = FALSE]
  )
}

prepare_halo_data <- function(
    admissions,
    patients,
    diagnoses,
    phecode_map,
    cross_sectional) {
  required_sets <- c("full", "train", "test")
  if (!is.list(cross_sectional) ||
      !all(required_sets %in% names(cross_sectional))) {
    stop("cross_sectional must contain full, train, and test data frames.")
  }

  phenotype_columns <- setdiff(
    names(cross_sectional$full),
    c("subject_id", "age80", "male", "white", "private")
  )
  visits <- .prepare_visits(admissions, patients, diagnoses, phecode_map) |>
    dplyr::filter(age >= 18)
  for (phenotype in setdiff(phenotype_columns, names(visits))) {
    visits[[phenotype]] <- 0L
  }

  full <- .encode_sensitive_attributes(
    visits,
    id_columns = c("subject_id", "hadm_id"),
    phenotype_columns = phenotype_columns
  )
  train_ids <- unique(cross_sectional$train$subject_id)

  list(
    full = full,
    train = dplyr::filter(full, subject_id %in% train_ids),
    test = dplyr::filter(full, !subject_id %in% train_ids)
  )
}

prepare_benchmark_data <- function(
    admissions,
    patients,
    diagnoses,
    phecode_map,
    min_phenotype_count = 20L,
    train_fraction = 0.8,
    split_seed = 123L) {
  cross_sectional <- prepare_cross_sectional_data(
    admissions = admissions,
    patients = patients,
    diagnoses = diagnoses,
    phecode_map = phecode_map,
    min_phenotype_count = min_phenotype_count,
    train_fraction = train_fraction,
    split_seed = split_seed
  )
  halo <- prepare_halo_data(
    admissions = admissions,
    patients = patients,
    diagnoses = diagnoses,
    phecode_map = phecode_map,
    cross_sectional = cross_sectional
  )
  list(cross_sectional = cross_sectional, halo = halo)
}
