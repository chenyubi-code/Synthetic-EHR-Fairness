# Validation and implementation notes

This organized copy contains source code, dependency specifications, documentation, licenses, and artificial-data tests. Original project files were read only. No clinical records, generated cohorts, trained models, cached study results, paper figures, or manuscript table-building scripts were copied.

## Checks completed for the manuscript update

| Area | Check | Result |
| :--- | :--- | :--- |
| Preprocessing | Artificial admissions, demographics, diagnoses, and mapping; character diagnosis codes; parent phenotypes; stratified split; consistent patient/visit partitions | Passed |
| SARE / MPPGE / MPCGE / MAPPE / MAPCE | Identical distributions, known prevalence shifts, preserved gaps with nonzero absolute errors, real-only screening, undefined correlations, missing subgroups | Passed |
| Downstream metrics | Ridge logistic regression, Random Forest, and XGBoost on artificial data; TSTR, TSRTR, and TRTS; fixed-model TRTR/TRTS identity | Passed |
| DSGE / DSAE | Analytical examples, independent aTPR reference-direction reductions, undefined-cell handling, reference validation | Passed |
| Privacy | Exact nearest Hamming distance and all tied neighbors against brute force; F1 inference; baseline signs; zero exposure; missing subgroup/candidate handling; repeatable random vote ties | Passed |
| Baseline generation | Resample support and seed reproducibility; SBS conditional marginals; matrix/schema roundtrip | Passed |
| VAE | One training epoch, saved model checkpoint, and binary sampling on artificial 347-column input | Passed |
| CorGAN | One autoencoder epoch, one adversarial epoch, saved final checkpoint, probability sampling and binary conversion | Passed |
| HALO | Artificial visit preparation, one training epoch, saved architecture restoration, autoregressive sampling, patient-level conversion | Passed |
| Frozen phenotype screen | Explicit primary-analysis phenotype names survive alternative subgroup recoding; identifiers are excluded from screening | Passed |
| Downstream summaries | Task/model and subgroup averaging within each generation repetition, then means and SEs across repetitions; known-value fixtures | Passed |
| Alternative definitions | ID-matched age85 and Black recoding, unchanged patient split/order/features, visit-level consistency, generator column-name preservation | Passed |
| Candidate-pool sensitivity | Exact batch Hamming kernel against brute force, shared member/nonmember pools, draw-level absolute gaps, within-seed averaging, five-seed SEs | Passed |
| Package contents | Source/configuration/license allowlist, Python and R parsing, Markdown code fences and local links, scan for original-machine paths | Passed |

These checks validate code behavior and workflow connections. They do not reproduce the study's numerical results or establish full-dataset training convergence.

## Checks requiring additional environments

- **External aTPR execution:** the reference-direction reduction and error formulas were tested, but the external `metrics-adjuster` package is not installed in the verification environment. Its external runtime was not executed for this update. `tests/test_atpr.R` is provided for running the integration check once the dependency is available.
- **medGAN training:** TensorFlow 1.x / Python 3.7 was not available in the local test environment. The retained implementation, environment file, and training/sampling commands were inspected; full runtime execution remains to be checked in that environment.
- **EHRDiff training/sampling:** the wrapper requires the separately obtained upstream modules and its Python 3.9 environment. The wrapper and fixed architecture/sampling settings were inspected; full runtime execution remains to be checked with those dependencies.
- **Full study runs:** no MIMIC data were accessed and no 1,000-epoch study training run was performed.

## Local test environment

- R 4.5.1; `data.table` 1.17.8, `Rcpp` 1.1.1.1.1, `glmnet` 4.1.10, `randomForest` 4.7.1.2, `xgboost` 3.2.1.1.
- Python 3.13.9; NumPy 2.3.5, pandas 2.3.3, PyTorch 2.12.0, scikit-learn 1.7.2.

The generation dependency files retain the original environment specifications. Passing these small checks on the local environment is additional evidence, not a replacement for validating the intended study environments.

## Manuscript-update scope

This update starts from upstream commit `4a9fc6ca644fd459271287f1fbbe81087e634b37`. That version already included MAPPE/MAPCE, DSAE, and baseline-adjusted privacy. The new changes add the candidate-pool size sensitivity analysis; fixed-phenotype support and ID-aligned recoding for age85/Black generator retraining; manuscript-level downstream summaries; explicit sensitive-column arguments for generators; and data/output exclusions. Existing primary-analysis defaults are retained.

The candidate-pool sensitivity kernel follows the original analysis implementation. For each repetition and attribute, it samples 20 matched pools, uses the identical pool for members and nonmembers, calculates each absolute subgroup gap first, and averages those gaps within repetition. Means and standard errors then use the independent generation repetitions. Undefined target F1 scores are not replaced with zeros or silently dropped; the matched analysis fails explicitly in that case.

RNG behavior for the three default downstream models is retained; selecting a subset or reordering models now preserves each model's benchmark seed. Alternative definitions use explicit `age85`/`black` names rather than silently changing the meaning of `age80`/`white`.

## Organization and adaptations

- The existing reusable preprocessing, training, and evaluation code provided the starting point. Within-subgroup phenotype errors, downstream absolute errors, and baseline-adjusted privacy contrasts were integrated according to the supplied manuscript and supporting analysis code.
- Training and generation are grouped in one directory. HALO retains one supporting subdirectory; all metric modules share one flat directory. Revision labels and manuscript-production scripts are absent.
- All public inputs and outputs are supplied as function arguments or command-line paths. No original-machine absolute paths are retained.
- Tabular input preparation now rejects missing or non-binary values instead of silently replacing them with zero. Valid binary study inputs retain the same values and column order.
- VAE training now saves a reusable checkpoint and exposes epoch/batch-size overrides while retaining study defaults. CorGAN saves the final epoch even outside its regular checkpoint interval. HALO sampling restores the architecture stored in its checkpoint.
- XGBoost thread selection falls back to one thread when the local environment cannot report its CPU count.
- Invalid metric cells remain explicit. Data-level and privacy summaries retain valid-count information; downstream summaries propagate missing contributing cells. See [metrics/README.md](metrics/README.md) for details.

Third-party license files and acknowledgments are retained under [generation/](generation/).
