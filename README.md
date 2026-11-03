# Synthetic EHR Subgroup Fairness Benchmark

**Do Synthetic Electronic Health Records Preserve Subgroup Differences? A Fairness-Oriented Benchmark of Generation Methods**

Code for training synthetic EHR generators and evaluating subgroup fidelity, downstream predictive performance, and privacy exposure using the methods in the accompanying manuscript.

[Workflow](#workflow) · [Repository structure](#repository-structure) · [Generation methods](#generation-methods) · [Evaluation metrics](#evaluation-metrics) · [Getting started](#getting-started) · [Data access](#data-access)

> This is a **code-only repository**. Patient records, generated cohorts, trained checkpoints, and cached results are not distributed here.

## Workflow

The benchmark asks whether synthetic data preserve differences observed between patient subgroups, and whether measured privacy exposure is balanced across those subgroups. Gap preservation, within-subgroup accuracy, and privacy exposure are evaluated as complementary properties.

```text
Externally obtained data + ICD-9-CM-to-PhecodeX mapping
                         |
                  preprocessing.R
                         |
             Fixed patient-level train/test split
                  /                       \
        Real training partition       Real test partition
                  |                       |
       Train generators separately   Train separate generators
       for each repetition           for proxy testing only
                  |                       |
       Synthetic training cohorts    Synthetic proxy test cohorts
                  \                       /
                     Metric evaluation
       Data fidelity | Downstream performance | Privacy
```

The study uses MIMIC-III v1.4: 37,442 adults, 343 binary parent-level PhecodeX indicators, and four binary sensitive attributes. The fixed 80:20 split contains 29,959 training patients and 7,483 test patients, stratified on the joint sensitive-attribute pattern. Each method uses five independent generations: 30,000 records per synthetic training cohort and 7,483 records per synthetic proxy test cohort.

## Repository structure

```text
Synthetic-EHR-Fairness/
├── README.md
├── preprocessing.R             # Optional cohort construction from external inputs
├── install_R_dependencies.R
├── requirements-atpr.txt
├── generation/                 # Generator training, sampling, and dependencies
├── metrics/                    # Reusable metric functions
│   ├── load.R                  # Loads all metric modules
│   ├── data_level.R            # SARE, MPPGE, MPCGE, MAPPE, MAPCE
│   ├── downstream.R            # Predictive evaluation, DSGE, and DSAE
│   ├── privacy.R               # Member/nonmember exposure and subgroup gaps
│   ├── classifiers.R           # Ridge logistic regression, RF, XGBoost
│   ├── atpr.R                  # Adjusted-TPR interface
│   ├── nearest_neighbor.R      # Hamming-distance and inference calculations
│   └── common.R                # Validation and phenotype screening
└── tests/                      # Tests using artificial data only
```

## Generation methods

| Method | Model or reference | Representation |
| :--- | :--- | :--- |
| Resample | Bootstrap resampling of complete source records | Patient-level binary table |
| SBS | Stratified Bernoulli sampling | Joint sensitive-attribute strata and conditional phenotype prevalences |
| VAE | Variational autoencoder | Patient-level binary table |
| medGAN | Generative adversarial network | Patient-level binary table |
| CorGAN | Correlation-aware GAN | Patient-level binary table |
| HALO | Autoregressive transformer | Visit sequences, aggregated to patient-level indicators after generation |
| EHRDiff | Diffusion model | Patient-level binary table |

Resample provides an empirical fidelity reference; it resamples source records and is not a privacy-preserving release mechanism. SBS reproduces subgroup-specific marginal prevalences while sampling phenotypes independently within each joint stratum.

See [generation/README.md](generation/README.md) for model-specific environments, training commands, sampling commands, and upstream dependencies. Each repetition should initialize and train a new model with its own seed.

## Evaluation metrics

| Domain | Metrics | What they measure |
| :--- | :--- | :--- |
| Sensitive-attribute representation | SARE | Error in marginal subgroup proportions |
| Between-subgroup phenotype structure | MPPGE, MPCGE | Errors in phenotype prevalence gaps and correlation gaps |
| Within-subgroup phenotype fidelity | MAPPE, MAPCE | Absolute prevalence and correlation errors within each subgroup |
| Downstream gap preservation | DSGE | Departure from the fixed real-data subgroup performance gap |
| Within-subgroup predictive agreement | DSAE | Absolute departure from each subgroup's fixed real-data performance |
| Within-subgroup privacy exposure | baMIR, baAIR | Member-versus-nonmember contrasts in nearest-record proximity and attribute inference |
| Privacy-risk parity | baMIRG, baAIRG | Absolute differences in baseline-adjusted exposure between subgroups |

Signed subgroup gaps always use **group 1 minus group 0**:

| Column | Group 0 | Group 1 |
| :--- | :--- | :--- |
| `age80` | Age <80 years | Age >=80 years |
| `male` | Female | Male |
| `white` | Non-White | White |
| `private` | Non-private insurance | Private insurance |

```text
Subgroup gap = metric(group 1) - metric(group 0)

DSGE = |synthetic-regime subgroup gap - TRTR subgroup gap|
DSAE = |synthetic-regime subgroup value - TRTR subgroup value|

baMIR = nearest-record distance(nonmembers) - nearest-record distance(members)
baAIR = attribute-inference F1(members) - attribute-inference F1(nonmembers)

baMIRG = |baMIR(group 1) - baMIR(group 0)|
baAIRG = |baAIR(group 1) - baAIR(group 0)|
```

Small gap errors do not establish accurate absolute performance within each subgroup. Similarly, small privacy gaps do not establish low privacy exposure. The member and nonmember diagnostics are retained for interpretation.

### Downstream settings

| Regime | Predictive-model training | Evaluation data | Purpose |
| :--- | :--- | :--- | :--- |
| TRTR | Real training data | Real test data | Fixed reference |
| TSTR | Synthetic training data | Same real test data | Training-data substitution |
| TSRTR | Real + synthetic training data | Same real test data | Data augmentation |
| TRTS | Same fixed real-trained model as TRTR | Separate synthetic proxy test data | Proxy testing |

The default outcomes are ischemic heart disease (`CV_404`) and renal failure (`GU_582`). Classifiers are ridge-penalized logistic regression, Random Forest, and XGBoost. Outcomes and classifiers can be selected in the evaluation function. Performance measures are AUC, ACC, TPR, adjusted TPR (aTPR), FPR, and PPR.

Sensitive attributes define evaluation subgroups and are excluded from the downstream predictors. The outcome phenotype is also excluded. Predictor screening uses the real training partition only: prevalence must exceed 1% in both groups of every sensitive attribute. Each default outcome must additionally have at least 50 positive and 50 negative real-test observations in every subgroup.

**TRTS uses separately generated test cohorts.** Its generators are trained on the real test partition solely to create proxy test data; the predictive model remains the fixed TRTR model trained on the real training partition. Privacy evaluation instead uses training-derived synthetic cohorts, with real training records as members and real test records as nonmembers.

## Getting started

### 1. Prepare the environments

R 4.1 or newer is needed for the preprocessing syntax. Install the R dependencies once:

```bash
Rscript install_R_dependencies.R
```

Privacy nearest-neighbor calculations also require a working C++ compiler. Generator environments are intentionally separate because their Python and deep-learning dependencies differ; use the instructions in [generation/README.md](generation/README.md).

For aTPR, install the external `metrics-adjuster` package in a **Python 3.11+ environment**:

```bash
python -m pip install -r requirements-atpr.txt
```

Use that environment's Python executable when evaluating downstream metrics:

```r
Sys.setenv(METRICS_ADJUSTER_PYTHON = "/path/to/atpr-environment/bin/python")
```

### 2. Prepare inputs

If the analysis-ready binary tables are already available, use them directly. Otherwise, the optional preprocessing functions accept four data frames obtained independently: `admissions`, `patients`, `diagnoses`, and `phecode_map`.

```r
source("preprocessing.R")

# These four input objects must already exist in your local R session.
# Read diagnosis and mapping codes as character values to retain leading zeros.
cohort <- prepare_benchmark_data(
  admissions = admissions,
  patients = patients,
  diagnoses = diagnoses,
  phecode_map = phecode_map
)

real_train <- cohort$cross_sectional$train
real_test <- cohort$cross_sectional$test
halo_train <- cohort$halo$train
halo_test <- cohort$halo$test
```

Preprocessing retains adults, removes inconsistent sensitive-attribute records, maps diagnoses to parent phenotypes, retains phenotypes recorded for at least 20 patients, and creates a fixed patient-level split. This minimum-count filter is distinct from the later 1% evaluation screen.

Metric inputs must contain complete binary 0/1 values, the four sensitive columns, and identically named phenotype columns. Optional `subject_id` and `hadm_id` columns are removed before evaluation. Synthetic inputs can be a single data frame or a named list of replicate data frames.

### 3. Train generators and evaluate cohorts

Follow the training commands in [generation/README.md](generation/README.md), then load your generated cohorts into R. The metric functions operate on the supplied objects and return results without building figures or manuscript tables.

```r
source("metrics/load.R")

# Named lists, one element per independent generation, loaded by the caller:
# synthetic_train <- list(seed_1 = ..., seed_2 = ..., ...)
# synthetic_test  <- list(seed_1 = ..., seed_2 = ..., ...)

data_metrics <- compute_data_level_metrics(
  real_data = real_train,
  synthetic_data = synthetic_train,
  method = "VAE"
)

downstream_metrics <- compute_downstream_metrics(
  real_train = real_train,
  real_test = real_test,
  synthetic_train = synthetic_train,
  synthetic_test = synthetic_test,
  method = "VAE",
  regimes = c("TSTR", "TSRTR", "TRTS")
)

privacy_metrics <- compute_privacy_metrics(
  real_train = real_train,
  real_test = real_test,
  synthetic_data = synthetic_train,
  method = "VAE"
)
```

When requesting both training and proxy-testing regimes, the two synthetic lists must have matching repetition names. Each proxy test cohort must contain the same number of rows as `real_test`.

See [metrics/README.md](metrics/README.md) for return objects, screening and missing-value conventions, and the reference-dependent aTPR calculation. `include_atpr = FALSE` explicitly runs downstream evaluation without aTPR; the default requires aTPR and fails clearly if its external dependency is unavailable.

### 4. Check the code with artificial data

```bash
Rscript tests/test_preprocessing.R
Rscript tests/test_data_level.R
Rscript tests/test_downstream.R
Rscript tests/test_privacy.R
python tests/test_generation.py

# Optional: requires the metrics-adjuster environment configured above
Rscript tests/test_atpr.R
```

These tests construct small artificial examples. They do not load MIMIC-III or require distributed patient records. See [VALIDATION.md](VALIDATION.md) for the checks performed on this organized copy and the remaining environment-dependent checks.

## Data access

MIMIC-III v1.4 must be obtained independently through [PhysioNet](https://physionet.org/content/mimiciii/1.4/), subject to its credentialing, training, and data-use requirements. The ICD-9-CM-to-PhecodeX mapping must also be obtained separately. No automatic clinical-data download is included.

Keep real data, generated patient-level cohorts, and checkpoints in a separate working location. The supplied `.gitignore` excludes common data and model-output formats from version control.

## Associated manuscript

Chenyu Bi, Weidong Ma, and Jinbo Chen. *Do Synthetic Electronic Health Records Preserve Subgroup Differences? A Fairness-Oriented Benchmark of Generation Methods.*

Source-method acknowledgments and third-party license notices are included under `generation/`.
