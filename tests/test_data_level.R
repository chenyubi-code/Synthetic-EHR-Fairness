# Artificial populations with analytically known prevalence/correlation behavior.
source("metrics/load.R")
library(data.table)
pattern <- expand.grid(age80 = 0:1, male = 0:1, white = 0:1, private = 0:1)
real <- pattern[rep(seq_len(nrow(pattern)), each = 8L), ]
real$P1 <- rep(c(0, 0, 1, 1, 0, 0, 1, 1), 16)
real$P2 <- rep(c(0, 1, 0, 1, 0, 1, 0, 1), 16)
identity <- compute_data_level_metrics(real, real)
stopifnot(all(as.matrix(identity$by_repetition[, .(SARE, MPPGE, MPCGE)]) == 0),
          all(as.matrix(identity$subgroup_by_repetition[, .(MAPPE, MAPCE)]) == 0))
# Both subgroups shift by the same amount: gap preserved, absolute fidelity changed.
shifted <- real
shifted$P1 <- rep(c(0, 1, 1, 1, 0, 1, 1, 1), 16)
changed <- compute_data_level_metrics(real, shifted)
stopifnot(all(abs(changed$by_repetition$MPPGE) < 1e-12),
          all(abs(changed$subgroup_by_repetition$MAPPE - .125) < 1e-12))
# A shift in age group 1 alone changes the signed prevalence gap by .25 for one phenotype.
one_group <- real
one_group$P1[one_group$age80 == 1] <- shifted$P1[one_group$age80 == 1]
known <- compute_data_level_metrics(real, one_group)
stopifnot(abs(known$by_repetition[sensitive_attribute == "age80", MPPGE] - .125) < 1e-12)
# Constant synthetic phenotypes yield undefined correlation, never a false zero error.
constant <- real
constant$P1 <- 1L
undefined <- compute_data_level_metrics(real, constant)
stopifnot(all(is.na(undefined$by_repetition$MPCGE)),
          all(is.na(undefined$subgroup_by_repetition$MAPCE)),
          all(undefined$subgroup_by_repetition$n_valid_correlation_pairs == 0L))
# Screening is derived from real data, even if the synthetic column becomes common.
real$rare <- 0L
shifted$rare <- 1L
screened <- compute_data_level_metrics(real, shifted)
stopifnot(!screened$phenotype_screen[phenotype == "rare", included])
identified <- real
identified$subject_id <- seq_len(nrow(identified))
stopifnot(identical(compute_eval_phenotypes(identified)$phenotypes,
                    compute_eval_phenotypes(real)$phenotypes))
table_input <- data.table::as.data.table(identified)
table_before <- data.table::copy(table_input)
stopifnot(identical(compute_eval_phenotypes(table_input)$phenotypes,
                    compute_eval_phenotypes(real)$phenotypes),
          identical(table_input, table_before))
# Absent generated subgroups are explicit missing cells.
missing_group <- compute_data_level_metrics(real, shifted[shifted$age80 == 0, ])
stopifnot(is.na(missing_group$subgroup_by_repetition[
  sensitive_attribute == "age80" & group_value == 1, MAPPE]))
# Alternative subgroup analyses keep the main benchmark's phenotype list, even
# when an individual phenotype fails the newly coded subgroup prevalence screen.
fixed <- compute_data_level_metrics(real, shifted, evaluation_phenotypes = c("P1", "rare"))
stopifnot(fixed$phenotype_screen[phenotype == "rare", included],
          !fixed$phenotype_screen[phenotype == "rare", passes_prevalence_screen],
          !fixed$phenotype_screen[phenotype == "P2", included],
          all(fixed$subgroup_by_repetition$n_eval_phenotypes == 2L),
          all(abs(fixed$subgroup_by_repetition$MAPPE - .625) < 1e-12),
          all(fixed$phenotype_screen$selection_rule == "supplied_fixed_phenotypes"))
for (invalid in list(c("P1", "P1"), c("P1", "missing"), c("P1", "age80"), NA_character_)) {
  stopifnot(inherits(try(compute_data_level_metrics(real, shifted,
    evaluation_phenotypes = invalid), silent = TRUE), "try-error"))
}
cat("PASS: all five data-level metrics, identity, known gap/absolute contrasts,\n",
    "real-only/fixed screening, constant features, and absent subgroups.\n")
