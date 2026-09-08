# Synthetic EHR Subgroup Fairness Benchmark

This repository contains the code used to preprocess the benchmark cohort,
generate synthetic electronic health records, and evaluate whether synthetic
data preserve differences across patient subgroups.

No patient-level data are included. The MIMIC-III data used in this study are
available through [PhysioNet](https://physionet.org/content/mimiciii/1.4/) to
credentialed researchers who complete the required training and data-use
agreement. Users must obtain MIMIC-III and the PhecodeX mapping independently.

## Repository structure

### `01_preprocessing`

- `preprocessing.R` contains in-memory functions that construct the adult
  patient-level dataset and the visit-level HALO representation from admissions,
  patient demographics, diagnoses, and the ICD-9-to-PhecodeX mapping. The
  functions create the four binary sensitive attributes, parent-level phenotype
  indicators, and the stratified training/test split.

### `02_generation`

Contains the implementations and configuration used to generate five synthetic
replicates with Resample, Stratified Bernoulli Sampling (SBS), VAE, medGAN,
CorGAN, HALO, and EHRDiff. See the folder README for model dependencies and
hyperparameters.

### `03_evaluation`

- `data_level.R` computes sensitive-attribute representation error (SARE), mean
  phenotype prevalence gap error (MPPGE), and mean phenotype correlation gap
  error (MPCGE).
- `downstream.R` evaluates TRTR, TSTR, TSRTR, and TRTS using ridge logistic
  regression, Random Forest, and XGBoost, and computes downstream subgroup gap
  error (DSGE).
- `privacy.R` computes membership-inference risk (MIR), attribute-inference risk
  (AIR), and their between-subgroup gaps.
- `lib` contains shared metric, classifier, adjusted-TPR, and nearest-neighbor
  functions.

Generated data, trained models, and analysis results are not included.
