# All rows are fabricated in memory; no clinical files are read.
source("preprocessing.R")
grid <- expand.grid(age80 = 0:1, male = 0:1, white = 0:1, private = 0:1)
grid <- grid[rep(seq_len(nrow(grid)), each = 5L), ]
n <- nrow(grid)
patients <- data.frame(subject_id = seq_len(n),
  gender = ifelse(grid$male, "M", "F"),
  dob = ifelse(grid$age80, "1930-01-01", "1980-01-01"))
admissions <- data.frame(subject_id = seq_len(n), hadm_id = seq_len(n) + 1000L,
  insurance = ifelse(grid$private, "Private", "Other"),
  ethnicity = ifelse(grid$white, "WHITE", "OTHER"), admittime = "2020-01-01")
diagnoses <- data.frame(subject_id = rep(seq_len(n), 2),
  hadm_id = rep(seq_len(n) + 1000L, 2), icd9_code = rep(c("0010", "0020"), each = n))
phecode_map <- data.frame(vocabulary_id = "ICD9CM", code = c("001.0", "002.0"),
  phecode = c("TOY_100.1", "TOY_200.2"))
result <- prepare_benchmark_data(admissions, patients, diagnoses, phecode_map)
stopifnot(nrow(result$cross_sectional$full) == 80L,
          nrow(result$cross_sectional$train) == 64L,
          nrow(result$cross_sectional$test) == 16L,
          !length(intersect(result$cross_sectional$train$subject_id,
                            result$cross_sectional$test$subject_id)),
          all(c("TOY_100", "TOY_200") %in% names(result$cross_sectional$full)),
          setequal(result$halo$train$subject_id, result$cross_sectional$train$subject_id),
          setequal(result$halo$test$subject_id, result$cross_sectional$test$subject_id))
cat("PASS: in-memory preprocessing, character diagnosis codes, parent phenotypes,\n",
    "joint stratification, and aligned patient/visit partitions.\n")
