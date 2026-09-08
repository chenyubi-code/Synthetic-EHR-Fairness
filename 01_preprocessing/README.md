# Preprocessing

`preprocessing.R` constructs the datasets used by the benchmark. It provides:

- `prepare_cross_sectional_data()` for the adult patient-level phenotype matrix
  and stratified training and test split;
- `prepare_halo_data()` for visit-level HALO inputs with the same phenotypes and
  patient split; and
- `prepare_benchmark_data()` for running both steps together.

The functions accept in-memory `admissions`, `patients`, `diagnoses`, and
PhecodeX mapping data frames. Diagnosis codes and mapping codes should be loaded
as character values to preserve leading zeros. Required columns are checked by
the code, and input column names are case-insensitive.

The pipeline removes patients with inconsistent recorded sex, ethnicity, or
insurance; calculates age at first admission; retains adults; maps ICD-9-CM
diagnoses to parent PhecodeX phenotypes; and retains phenotypes recorded for at
least 20 patients. It encodes age 80 or older, male sex, White ethnicity, and
private insurance, then makes an 80:20 split stratified by the joint attribute
pattern with seed 123.

Each returned object contains `full`, `train`, and `test` data frames. Nothing is
written to disk. Required R packages are `dplyr`, `tidyr`, `stringr`, `icd`, and
`caret`. Clinical data and the PhecodeX mapping are not included.
