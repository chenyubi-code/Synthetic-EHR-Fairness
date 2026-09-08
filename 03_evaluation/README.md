# Benchmark evaluation

This directory contains the functions used to evaluate binary, cross-sectional synthetic EHR data. The functions accept data frames, compute the metrics in memory, and return metric tables without saving data or results.

## Files

| File | Purpose |
| --- | --- |
| `data_level.R` | Computes sensitive-attribute representation error (SARE), mean phenotype prevalence-gap error (MPPGE), and mean phenotype correlation-gap error (MPCGE). |
| `downstream.R` | Fits ridge logistic regression, random forest, and XGBoost models; evaluates TRTR, TSTR, TSRTR, and TRTS; and computes subgroup metrics and DSGE. |
| `privacy.R` | Computes membership inference risk (MIR), attribute inference risk (AIR), and their subgroup gaps (MIRG and AIRG). |
| `lib/common.R` | Validates inputs, selects evaluation phenotypes, and summarizes repetitions. |
| `lib/classifiers.R` | Defines classifier settings and prediction metrics. |
| `lib/atpr.R` | Computes adjusted true-positive rate (aTPR) with `metrics-adjuster`. |
| `lib/nearest_neighbor.R` | Implements exact Hamming nearest-neighbor search with tied-neighbor voting for privacy metrics. |

The main functions are `compute_data_level_metrics()`, `compute_downstream_metrics()`, and `compute_privacy_metrics()`. A synthetic input can be one `data.frame` or a named list of `data.frame` objects, with one list element per repetition.

## Input data

Each data frame must contain complete binary 0/1 values. The four sensitive attributes are `age80`, `male`, `white`, and `private`; all remaining analysis columns are phenotype indicators. The optional identifier columns `subject_id` and `hadm_id` are removed before evaluation. Real and synthetic inputs must use the same analysis-column names. Synthetic test data used for TRTS must have the same number of rows as the real test data.

Phenotypes are included in the data-level and downstream evaluations when their prevalence is greater than 1% in both groups of every sensitive attribute in the real training data. Downstream outcomes must also have at least 50 positive and 50 negative real-test observations in every subgroup. Privacy metrics use all phenotype columns.

## Dependencies

- Data-level metrics: R and `data.table`.
- Downstream metrics: `data.table`, `Matrix`, `glmnet`, `pROC`, `randomForest`, `xgboost`, and Python 3.11 or newer with `metrics-adjuster`.
- Privacy metrics: `data.table`, `Rcpp`, and a C++ compiler.
