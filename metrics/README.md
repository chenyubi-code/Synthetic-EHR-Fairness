# Metric functions and conventions

Load the modules with `source("metrics/load.R")`. Inputs stay in memory; results are returned as named lists of tables. The aTPR adapter uses an automatically cleaned temporary directory to communicate with its external Python implementation.

## Common input contract

- Every analysis column contains complete binary 0/1 values. Numeric, logical, and character/factor representations of 0/1 are accepted.
- The default sensitive columns are `age80`, `male`, `white`, and `private`.
- All other analysis columns are phenotype indicators. Optional `subject_id` and `hadm_id` are removed.
- Real and synthetic tables must have the same analysis-column names; synthetic columns are reordered to match the reference.
- A synthetic input may be one data frame or a named list of data frames. List names identify independent repetitions and must be unique and non-empty.

The functions calculate per-repetition quantities before averaging repetitions. A mean of absolute errors is generally different from the absolute error of a mean; the latter is not substituted for the former.

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

Screen phenotypes using the **real training data only**: prevalence must be strictly greater than the threshold in both subgroups of every sensitive attribute. SARE is computed from subgroup proportions; the four phenotype metrics use the screened set.

Correlation metrics use Pearson correlations for distinct unordered phenotype pairs, taken once from the upper triangle, excluding the diagonal. At least two phenotypes must pass screening. Correlations are undefined for constant columns, and the retained implementation requires at least three records in a subgroup. Undefined comparisons remain `NA`, are omitted from finite-pair means, and are counted explicitly. A cell with no valid comparisons remains `NA`; undefined correlation is never replaced with zero.

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

The main evaluator's optional `summary` averages DSGE within each repetition and classifier before averaging repetitions. Undefined contributing cells propagate to that summary, with coverage counts. The unaggregated DSGE/DSAE tables remain the primary reusable outputs; no manuscript-specific table formatting or plotting is performed.

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
