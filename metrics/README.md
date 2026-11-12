# Metric functions and conventions

Load the modules with `source("metrics/load.R")`. Inputs stay in memory; results are returned as named lists of tables. The aTPR adapter uses an automatically cleaned temporary directory to communicate with its external Python implementation.

## Common input contract

- Every analysis column contains complete binary 0/1 values. Numeric, logical, and character/factor representations of 0/1 are accepted.
- The default sensitive columns are `age80`, `male`, `white`, and `private`.
- All other analysis columns are phenotype indicators. Optional `subject_id` and `hadm_id` are removed.
- Real and synthetic tables must have the same analysis-column names; synthetic columns are reordered to match the reference.
- A synthetic input may be one data frame or a named list of data frames. List names identify independent repetitions and must be unique and non-empty.

The functions calculate per-repetition quantities before averaging repetitions. A mean of absolute errors is generally different from the absolute error of a mean; the latter is not substituted for the former. The study uses five generator repetitions (seeds `202601`–`202605`). Standard errors summarize variability across independent generator repetitions, not across patients, outcomes, or individual privacy subsamples.

## Data-level fidelity

```r
result <- compute_data_level_metrics(
  real_data = real_train,
  synthetic_data = synthetic_train,
  method = "VAE",
  prevalence_threshold = 0.01
)

result$by_repetition            # SARE, MPPGE, MPCGE by attribute and repetition
result$subgroup_by_repetition   # MAPPE, MAPCE by subgroup and repetition
result$phenotype_screen         # Real-training prevalence filter
```

| Metric | Full name | Definition |
| :--- | :--- | :--- |
| SARE | Sensitive-Attribute Representation Error | Absolute difference in the proportion belonging to group 1 |
| MPPGE | Mean Phenotype Prevalence Gap Error | Mean absolute error in group-1-minus-group-0 phenotype prevalence gaps |
| MPCGE | Mean Phenotype Correlation Gap Error | Mean absolute error in group-1-minus-group-0 phenotype correlation gaps |
| MAPPE | Mean Absolute Phenotype Prevalence Error | Mean absolute real-versus-synthetic prevalence difference within one subgroup |
| MAPCE | Mean Absolute Phenotype Correlation Error | Mean absolute real-versus-synthetic correlation difference within one subgroup |

By default, screen phenotypes using the **real training data only**: prevalence must be strictly greater than the threshold in both subgroups of every sensitive attribute. SARE is computed from subgroup proportions; the four phenotype metrics use the screened set. The study screen retained 92 phenotypes; correlation metrics use the 4,186 distinct pairs.

For an alternative subgroup definition, freeze the main-analysis phenotype set with `evaluation_phenotypes` rather than screening again:

```r
main_screen <- compute_eval_phenotypes(real_train)
fixed_phenotypes <- main_screen$phenotypes

# recoded_train and recoded_synthetic_train must use the alternative coding.
result <- compute_data_level_metrics(
  real_data = recoded_train,
  synthetic_data = recoded_synthetic_train,
  method = "VAE",
  sensitive_columns = c("age85", "male", "white", "private"),
  evaluation_phenotypes = fixed_phenotypes
)
```

`evaluation_phenotypes = NULL` retains automatic screening. An explicit vector must contain unique phenotype names present in the input. The returned `phenotype_screen` distinguishes the current `passes_prevalence_screen` diagnostic from the actual inclusion decision and records `selection_rule`; an explicitly fixed phenotype is not removed merely because prevalence changes under recoding. The same override is accepted by `compute_downstream_metrics()`.

Correlation metrics use Pearson correlations for distinct unordered phenotype pairs, taken once from the upper triangle, excluding the diagonal. At least two phenotypes must be selected. Correlations are undefined for constant columns, and the retained implementation requires at least three records in a subgroup. Undefined comparisons remain `NA`, are omitted from finite-pair means, and are counted explicitly. A cell with no valid comparisons remains `NA`; undefined correlation is never replaced with zero.

Additional outputs `summary` and `subgroup_summary` give per-attribute/per-subgroup means and standard errors across repetitions. Their `*_n_valid` and `n_repetitions` columns make partial coverage explicit. Detailed tables also retain the number of evaluated phenotypes and valid correlation pairs.

## Downstream performance, DSGE, and DSAE

```r
result <- compute_downstream_metrics(
  real_train = real_train,
  real_test = real_test,
  synthetic_train = synthetic_train,
  synthetic_test = synthetic_test,
  method = "VAE",
  regimes = c("TSTR", "TSRTR", "TRTS"),
  outcomes = c("CV_404", "GU_582"),
  models = c("ridge_logistic", "random_forest", "xgboost")
)

result$subgroup_metrics          # TRTR and each requested synthetic regime
result$dsge_by_repetition        # Gap errors by task/model/attribute/repetition
result$dsae_by_repetition        # Absolute errors with subgroup retained
result$dsae_reference_details    # Both raw aTPR reference-direction errors
result$predictors               # Screened predictors for each outcome
```

Use `evaluation_phenotypes = fixed_phenotypes` and the matching `sensitive_columns` when running downstream sensitivity analyses. The outcome is excluded from its own predictors, and the sensitive columns are excluded from all predictor sets. By default, an outcome must have at least 50 positive and 50 negative real-test observations in every subgroup; the `minimum_test_cases` argument exposes this eligibility threshold.

The function fits the real-data reference once for each outcome and classifier. TRTS reuses that fitted model. TSTR and TSRTR fit new predictive models for each synthetic repetition. Synthetic test tables must have the same number of rows as the real test table, and paired synthetic lists must have matching repetition names.

The retained model defaults are ridge penalty 0.05; 100 Random Forest trees with terminal-node size 10; and 100 XGBoost rounds with depth 3 and learning rate 0.05. The classification threshold is 0.5. Full settings are in [classifiers.R](classifiers.R).

```text
G = metric(group 1) - metric(group 0)
DSGE = |G_synthetic_regime - G_TRTR|
DSAE(group a) = |metric_synthetic_regime(group a) - metric_TRTR(group a)|
```

The reported metrics are AUC, ACC, TPR, FPR, PPR, and aTPR. AUC requires both outcome classes; TPR and FPR require their respective denominators. Undefined values remain `NA`.

### aTPR reference directions

aTPR is computed through `metrics-adjuster`, preserving the original calibration degree 2, density-ratio degree 1, and analysis seed. It is evaluated twice: once with subgroup 0 as the risk-distribution reference and once with subgroup 1 as the reference. The reference subgroup retains its ordinary TPR.

- **aTPR DSGE:** calculate the subgroup gap error under each reference, then retain the larger error for that task/model/repetition/attribute cell.
- **aTPR DSAE:** calculate each subgroup's absolute error under each reference, then retain that subgroup's larger error. The two subgroups can select different references, and their choices need not match the DSGE choice.
- Both reference-specific columns are retained in `subgroup_metrics`; DSAE also returns the full reference-specific comparison table.
- Both reference-specific errors must be finite for the reduced error to be valid.

`include_atpr = FALSE` explicitly excludes aTPR and requires no Python adapter. It leaves the other five downstream metrics available. The default is `TRUE`; missing or failing external aTPR execution produces an error rather than silently substituting ordinary TPR.

### Evaluating existing subgroup scores

If classifier results are already available **in memory**, the mathematical error functions can be called without refitting models:

```r
gap_errors <- compute_dsge(subgroup_metrics)
absolute_errors <- compute_dsae(subgroup_metrics)
```

Required identifiers are `method`, `repetition`, `regime`, `prediction_model`, `phenotype`, `sensitive_attribute`, and `group_value`, together with `AUC`, `ACC`, `TPR`, `FPR`, and `PPR`. Supply either both `aTPR_ref0`/`aTPR_ref1` columns or neither. Include exactly one fixed TRTR reference for each model/outcome/attribute/subgroup; it is shared across synthetic repetitions. Duplicate or incomplete reference cells are rejected.

The existing `summary` retains model-specific DSGE averages. The additional summaries follow the manuscript's equal-weight aggregation:

| Output | Aggregation within each repetition |
| :--- | :--- |
| `dsge_summary_by_attribute` | Average tasks and classifier classes, retaining each sensitive-attribute contrast |
| `dsge_summary` | Average tasks, classifier classes, and sensitive-attribute contrasts |
| `dsae_summary_by_subgroup` | Average tasks and classifier classes, retaining each marginal subgroup |
| `dsae_summary` | Average tasks, classifier classes, and marginal subgroups |

The repetition-level outputs `dsge_attribute_by_repetition`, `dsge_overall_by_repetition`, `dsae_subgroup_by_repetition`, and `dsae_overall_by_repetition` retain the values used for the outer means and standard errors. With the study defaults, downstream means combine two tasks and three classifiers, and overall means equally weight four contrasts or eight marginal subgroups. Undefined contributing cells propagate to these summaries, with coverage counts. The unaggregated DSGE/DSAE tables remain available; no manuscript-specific table formatting or plotting is performed.

## Privacy exposure and parity

```r
result <- compute_privacy_metrics(
  real_train = real_train,
  real_test = real_test,
  synthetic_data = synthetic_train,
  method = "VAE"
)

result$population_metrics   # MIR/AIR separately for members and nonmembers
result$by_repetition        # Paired subgroup exposures: baMIR, baAIR
result$gaps                 # baMIRG, baAIRG within each repetition
result$target_metrics       # F1/accuracy for each inferred sensitive attribute
```

Only synthetic cohorts generated from the **real training partition** are used here. Real training records are members (`mem`), and held-out real test records are nonmembers (`nonmem`). Both populations are evaluated against the same synthetic candidate pool within each subgroup and repetition. Population-level scores are paired; individual training and test records are not matched to each other.

- **MIR:** mean minimum Hamming distance across all phenotype columns to synthetic candidates in the same sensitive-attribute subgroup. Distances are counts of differing binary entries. `MIR_normalized` divides by the number of phenotype columns.
- **AIR:** an attacker knows the phenotype vector and one sensitive attribute. For each of the other attributes, prediction uses the majority label among all equally nearest synthetic records in the matching known-attribute subgroup. AIR averages the resulting binary F1 scores over the three unknown attributes; accuracy is returned as an additional diagnostic.
- **Exact voting ties:** use the retained deterministic seed construction. Nonmember tie-breaking uses a distinct `|nonmem` seed label. A zero adjusted score is therefore expected for equal population scores, but identical finite samples with random vote ties need not produce exactly identical F1 scores under the two random streams.

```text
baMIR = MIR_nonmember - MIR_member
baAIR = AIR_member - AIR_nonmember
baMIRG = |baMIR_group1 - baMIR_group0|
baAIRG = |baAIR_group1 - baAIR_group0|
```

Positive baMIR indicates closer synthetic matches for members; positive baAIR indicates better attribute inference for members. Signed adjusted exposures are retained, including negative values. Privacy parity gaps are computed **within each repetition before averaging**.

Privacy uses every phenotype column, without the data-level/downstream prevalence screen. Empty real subgroups, empty synthetic candidate pools, and undefined F1 denominators remain explicitly missing. The output retains subgroup sizes, candidate counts, valid target counts, and validity flags. Compact `summary` and `subgroup_summary` outputs use finite repetition values with valid-count columns.

The exact Hamming search is implemented through Rcpp and is tested against an independent brute-force calculation. Loading the R modules does not compile or run an attack; compilation occurs on the first privacy calculation. No attack records or nearest-neighbor patient details are written by these functions.

## Matched synthetic candidate pools

Supplementary Methods S1.5 evaluates whether unequal synthetic subgroup sizes affect privacy-risk parity. The matched evaluator reuses the exact Hamming/tied-neighbor attack and changes only the available synthetic candidate pool:

```r
matched <- compute_matched_privacy_metrics(
  real_train = real_train,
  real_test = real_test,
  synthetic_data = synthetic_train,
  method = "VAE",
  n_resamples = 20L
)

matched$pool_design             # Full and matched candidate counts
matched$population_metrics      # Full pool and each matched-pool draw
matched$by_resample             # Signed baMIR/baAIR for each subgroup and draw
matched$gaps_by_resample         # Absolute gaps computed separately per draw
matched$by_repetition           # Subgroup exposures averaged within repetition
matched$gaps_by_repetition      # Gap values averaged within repetition
matched$summary                # Gap mean/SE across independent repetitions
matched$subgroup_summary        # Subgroup exposure mean/SE across repetitions
```

For each attribute and synthetic repetition, set `m = min(n_group0, n_group1)`. Keep the smaller subgroup intact and sample `m` candidates without replacement from the larger subgroup. Real members and nonmembers use the **same sampled synthetic pool** within each subgroup. The real populations and trained generators are unchanged.

The full-pool condition uses the original candidates (`resample = 0`); positive resample indices identify matched draws. `n_resamples` accepts 1–31, with 20 reproducing the study design. For each draw, calculate signed subgroup exposures and their absolute between-subgroup gaps. Average those **gap values** across draws within a generator repetition, then calculate the mean and standard error from the five repetition-level means. The 100 total draws are not treated as 100 independent generator repetitions. `condition` distinguishes `full` and `matched` results.

Both real populations and the synthetic data must contain both subgroups for each requested attribute; missing subgroups produce an explicit error in this sensitivity analysis. An undefined inferred-target F1 also produces an explicit error, preserving the complete three-target attack definition in every matched draw.

Exact batched neighbor search is implemented in [batch_neighbors.cpp](batch_neighbors.cpp). As with the primary privacy API, the caller supplies authorized data locally and the function returns metric summaries rather than patient-level nearest-neighbor outputs.

## Alternative subgroup definitions

Supplementary Methods S1.6 changes one sensitive definition at a time: age >=85 versus <85, or Black versus non-Black, where the recorded ethnicity label begins with `BLACK`. The cohort, original train/test membership, all phenotype columns, and main-analysis evaluation phenotype set must remain fixed.

The preparation helper in [../subgroup_sensitivity.R](../subgroup_sensitivity.R) uses locally supplied raw age/ethnicity metadata to recode the existing partitions. It does not infer raw attributes from the main-analysis binary flags and does not generate synthetic data. Retrain each learned generator and regenerate both baselines under the alternative coding, including separate real-test-source TRTS cohorts, before passing the recoded real and synthetic frames to the same evaluation APIs.

Prepare the fixed partitions as follows:

```r
source("subgroup_sensitivity.R")
fixed_phenotypes <- compute_eval_phenotypes(real_train)$phenotypes

# raw_attributes contains one row per subject_id and the original age and/or
# ethnicity column. age is age at the first recorded admission.
age85_inputs <- prepare_subgroup_sensitivity(
  real_train = real_train,
  real_test = real_test,
  raw_attributes = raw_attributes,
  evaluation_phenotypes = fixed_phenotypes,
  scenario = "age85"
)

# Retrain/generate externally using age85_inputs$real_train and $real_test.
# Then load those new synthetic cohorts as age85_synthetic_train/_test.
age85_data_metrics <- compute_data_level_metrics(
  real_data = age85_inputs$real_train,
  synthetic_data = age85_synthetic_train,
  method = "VAE",
  sensitive_columns = age85_inputs$sensitive_columns,
  evaluation_phenotypes = age85_inputs$evaluation_phenotypes
)
```

`prepare_subgroup_sensitivity()` checks patient-level uniqueness and disjoint train/test IDs, retains row order and all unchanged values, and replaces `age80` with `age85` or `white` with `black`. It checks that the original binary coding agrees with the raw source attribute. Use `scenario = "black"` for the race analysis. The returned list contains `real_train`, `real_test`, `sensitive_columns`, `evaluation_phenotypes`, and `scenario`.

For HALO visit tables, call `recode_sensitive_definition(halo_train, raw_attributes, "age85")` and the corresponding test-table call, retaining `subject_id` and `hadm_id`. Apply recoding to real inputs before fitting new generators; do not relabel already-generated synthetic records.

The baseline, VAE, and matrix-preparation commands accept `--sensitive-columns age85 male white private` or `--sensitive-columns age80 male black private`. medGAN, CorGAN, and EHRDiff consume the prepared matrices and do not take this flag. HALO dataset preparation uses the corresponding `--label_cols` arguments. Matrix-based generators retain the resulting feature order; restore the supplied feature names during conversion. See [generation examples](../generation/README.md) for complete commands.

Use explicit sensitive-column names for the alternative definitions and pass `evaluation_phenotypes = fixed_phenotypes` to both data-level and downstream evaluations. Privacy uses all phenotype columns and needs only the corresponding sensitive-column names. Do not run the original cohort splitter again or reselect evaluation phenotypes after changing the subgroup definition.
