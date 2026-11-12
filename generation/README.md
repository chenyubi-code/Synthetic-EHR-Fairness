# Generator training and sampling

Run commands from the repository root. All input and output locations are supplied by the caller. The scripts contain no patient data or pretrained models.

## Environments

| Methods | Environment | Dependency file |
| :--- | :--- | :--- |
| Resample, SBS, matrix preparation/conversion | Python with NumPy and pandas | Included in the PyTorch environment below |
| VAE, CorGAN, HALO | Python 3.10, PyTorch | `generation/requirements-torch.txt` |
| medGAN | Python 3.7, TensorFlow 1.x, Linux x86_64 | `generation/requirements-medgan.txt` |
| EHRDiff | Python 3.9, PyTorch 1.11 | `generation/requirements-ehrdiff.txt` |

Install a dependency file in its matching environment; these are separate environments, not a combined requirements list. For example, with Python 3.10 active:

```bash
python -m pip install -r generation/requirements-torch.txt
```

medGAN retains its original TensorFlow 1.x implementation and should use the listed Linux environment. EHRDiff additionally requires the upstream code described below. The dependency files preserve the source project's environment specifications; the small local smoke checks are documented in [../VALIDATION.md](../VALIDATION.md).

## Inputs and repetition settings

Patient-level tables contain `age80`, `male`, `white`, `private`, and binary phenotype columns. Optional `subject_id` and `hadm_id` are removed before tabular generation. HALO instead uses visit-level tables that include both identifiers.

Set these paths to your own external working location. The example paths below are placeholders, not bundled files:

```bash
INPUT_CSV="/path/to/authorized/patient_train.csv"
VISITS_CSV="/path/to/authorized/visit_train.csv"
RUN_DIR="/path/to/working/seed_1"
SEED=202601
N_GENERATE=30000
mkdir -p "$RUN_DIR"
```

Repeat training from a new initialization for each of the five seeds (`202601` through `202605`), using a different output directory for every method and repetition. To create TRTS cohorts, use the separately prepared **real test partition** as the generator input, train new generator instances, and set `N_GENERATE` to the real test row count (7,483 in the study). Predictive models are never trained on that real test partition.

## Resample and SBS

```bash
python generation/baselines.py --method Resample \
  --input-csv "$INPUT_CSV" --output-csv "$RUN_DIR/resample.csv" \
  --seed "$SEED" --n-generate "$N_GENERATE"

python generation/baselines.py --method SBS \
  --input-csv "$INPUT_CSV" --output-csv "$RUN_DIR/sbs.csv" \
  --seed "$SEED" --n-generate "$N_GENERATE"
```

Both baselines directly generate records and have no trained checkpoint. The same implementations are available as `generate_resample()` and `generate_sbs()` for data frames already in memory.

## VAE

```bash
python generation/vae.py \
  --input-csv "$INPUT_CSV" --output-csv "$RUN_DIR/vae.csv" \
  --model-path "$RUN_DIR/vae.pt" \
  --seed "$SEED" --n-generate "$N_GENERATE"
```

The default uses 1,000 epochs, a 64-dimensional latent space, hidden layers of 512 and 256 units, and a batch size of 1,024. The checkpoint includes the fitted weights, feature names, and architecture settings. `--epochs` and `--batch-size` can be reduced for a short trial; the defaults retain the study settings.

## Shared binary matrix preparation

medGAN and CorGAN consume NumPy matrices. Save the exact feature order alongside the matrix:

```bash
python generation/prepare_binary_matrix.py \
  --input-csv "$INPUT_CSV" --output-npy "$RUN_DIR/train.npy" \
  --columns-json "$RUN_DIR/columns.json"
```

The helper validates complete binary inputs. It does not silently impute missing values or convert invalid entries into zeros. The feature-name file allows generated matrices to be converted back to correctly labeled tables.

## medGAN

Activate the Python 3.7 / TensorFlow 1.x environment, then train:

```bash
mkdir -p "$RUN_DIR/medgan"
python generation/medgan.py "$RUN_DIR/train.npy" "$RUN_DIR/medgan/model" \
  --seed "$SEED" --n_pretrain_epoch 100 --n_epoch 1000 --save_max_keep 1

python generation/medgan.py "$RUN_DIR/train.npy" "$RUN_DIR/medgan/probabilities.npy" \
  --generate_data true --model_file "$RUN_DIR/medgan/model-999" \
  --seed "$SEED" --n_samples "$N_GENERATE"
```

Checkpoint epoch suffixes are zero-based; the default final checkpoint is `model-999`. If the training epoch count changes, use the checkpoint path printed by training. The generated NumPy array contains decoded probabilities. Convert them using Bernoulli sampling:

```bash
python generation/matrix_to_csv.py \
  --input-npy "$RUN_DIR/medgan/probabilities.npy" \
  --columns-json "$RUN_DIR/columns.json" --output-csv "$RUN_DIR/medgan.csv" \
  --probabilities --seed "$SEED"
```

## CorGAN

Activate the PyTorch environment:

```bash
python generation/corgan.py \
  --DATASETPATH "$RUN_DIR/train.npy" --expPATH "$RUN_DIR/corgan" \
  --manual_seed "$SEED" --training true --generate true \
  --n_epochs_pretrain 100 --n_epochs 1000 --n_generate "$N_GENERATE" --cuda false

python generation/matrix_to_csv.py \
  --input-npy "$RUN_DIR/corgan/synthetic_prob.npy" \
  --columns-json "$RUN_DIR/columns.json" --output-csv "$RUN_DIR/corgan.csv" \
  --probabilities --seed "$SEED"
```

Use `--cuda true` when a suitable CUDA device is available. The final checkpoint is saved even when the chosen epoch count is not a multiple of the checkpoint interval. The matrix must have at least `--batch_size` rows; reduce this setting for a small artificial-data trial.

## HALO

HALO uses visit-level inputs from `cohort$halo$train` (or the corresponding test partition when generating TRTS cohorts):

```bash
python generation/halo/build_halo_dataset_repeat.py \
  --input_csv "$VISITS_CSV" --out_dir "$RUN_DIR/halo_data"

python generation/halo/train_halo_repeat.py \
  --data_dir "$RUN_DIR/halo_data" --save_dir "$RUN_DIR/halo_model" --seed "$SEED"

python generation/halo/generate_halo_repeat.py \
  --data_dir "$RUN_DIR/halo_data" --model_path "$RUN_DIR/halo_model/halo_model" \
  --out_pkl "$RUN_DIR/halo.pkl" --seed "$((SEED + 1000))" --n_generate "$N_GENERATE"

python generation/halo/convert_halo_repeat.py \
  --data_dir "$RUN_DIR/halo_data" --synthetic_pkl "$RUN_DIR/halo.pkl" \
  --output_csv "$RUN_DIR/halo.csv"
```

The training script retains the best validation checkpoint. Sampling restores the saved architecture settings. The final conversion aggregates generated visits into the common patient-level phenotype representation. These generated intermediates remain in the external working location.

## EHRDiff

The wrapper uses the authors' upstream implementation at commit `26f064df711735af5d6c0d746dfd71ef2672c152`. Obtain that code separately:

```bash
EHRDIFF_SOURCE="/path/to/upstream/EHRDiff"
git clone https://github.com/sczzz3/EHRDiff.git "$EHRDIFF_SOURCE"
git -C "$EHRDIFF_SOURCE" checkout 26f064df711735af5d6c0d746dfd71ef2672c152
```

`EHRDIFF_SOURCE` must point to the directory containing `model/`, `denoiser.py`, `score_losses.py`, `samplers.py`, and `utils/`. Use the supplied dependency file in its Python 3.9 environment. Only upstream code is needed; obtain clinical inputs independently through the official data source.

Prepare 347 binary features plus the all-zero padding column required by the study architecture:

```bash
python generation/prepare_binary_matrix.py \
  --input-csv "$INPUT_CSV" --output-npy "$RUN_DIR/train_padded.npy" \
  --columns-json "$RUN_DIR/columns_padded.json" --pad-ehrdiff

python generation/ehrdiff.py --mode train \
  --source "$EHRDIFF_SOURCE" --matrix "$RUN_DIR/train_padded.npy" \
  --model-path "$RUN_DIR/ehrdiff.pt" --output-npy "$RUN_DIR/ehrdiff.npy" \
  --seed "$SEED" --n-generate "$N_GENERATE"

python generation/ehrdiff.py --mode generate \
  --source "$EHRDIFF_SOURCE" --matrix "$RUN_DIR/train_padded.npy" \
  --model-path "$RUN_DIR/ehrdiff.pt" --output-npy "$RUN_DIR/ehrdiff.npy" \
  --seed "$SEED" --n-generate "$N_GENERATE"

python generation/matrix_to_csv.py \
  --input-npy "$RUN_DIR/ehrdiff.npy" --columns-json "$RUN_DIR/columns_padded.json" \
  --output-csv "$RUN_DIR/ehrdiff.csv"
```

The wrapper retains CPU training, 1,000 epochs, the final EMA checkpoint, and the deterministic 32-step Heun EDM sampler. Sampling creates the original 30,000-row batch and retains the requested rows; `--n-generate` accepts 1 through 30,000. The conversion removes the padding column. Parallel runs must use different `--port` values.

## Alternative sensitive-attribute definitions

Supplementary Methods S1.6 retrains every generator for age >=85 years and,
separately, Black versus non-Black race. Use `prepare_subgroup_sensitivity()`
from [subgroup_sensitivity.R](../subgroup_sensitivity.R) on the original
patient-level train/test partitions; use `recode_sensitive_definition()` on
the corresponding HALO visit partitions. Keep all 343 generation phenotypes; the fixed 92
evaluation phenotypes are supplied to the evaluation functions only. Keep
each scenario, partition, and seed in a separate output directory.

The helpers use explicit names (`age85` or `black`), so pass the corresponding
four-column schema. For example, after exporting the age85 training partition
locally:

```bash
python generation/baselines.py --method SBS \
  --input-csv "$INPUT_CSV" --output-csv "$RUN_DIR/sbs.csv" \
  --seed "$SEED" --n-generate "$N_GENERATE" \
  --sensitive-columns age85 male white private

python generation/prepare_binary_matrix.py \
  --input-csv "$INPUT_CSV" --output-npy "$RUN_DIR/train.npy" \
  --columns-json "$RUN_DIR/columns.json" \
  --sensitive-columns age85 male white private

python generation/vae.py \
  --input-csv "$INPUT_CSV" --output-csv "$RUN_DIR/vae.csv" \
  --seed "$SEED" --n-generate "$N_GENERATE" \
  --sensitive-columns age85 male white private

python generation/halo/build_halo_dataset_repeat.py \
  --input_csv "$VISITS_CSV" --out_dir "$RUN_DIR/halo_data" \
  --label_cols age85 male white private
```

For the Black scenario use `age80 male black private` instead. Resample accepts
the same `--sensitive-columns` option as SBS. medGAN and CorGAN consume the
prepared matrices; their commands do not require label-name flags. EHRDiff
also consumes a matrix: add `--pad-ehrdiff` during preparation and retain its
matching column manifest. HALO training, sampling, and conversion restore the
label names from the scenario-specific prepared directory. Refit both the
training-derived and test-derived generators across all five seeds; do not
relabel already-generated primary-analysis cohorts.

The existing model defaults are retained. In medGAN, the GAN batch size is
1,000 and the autoencoder pretraining batch size is 100; these are separate
training stages.

## Model attribution

See [THIRD_PARTY.md](THIRD_PARTY.md), [LICENSE-medGAN](LICENSE-medGAN), and [LICENSE-HALO](LICENSE-HALO) for the retained source-method notices.
