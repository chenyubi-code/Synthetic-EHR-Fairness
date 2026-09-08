# Synthetic cohort generation

This folder contains the seven synthetic-data methods used in the benchmarking evaluation. Patient-level data, trained models, and generated cohorts are not included.

## Contents

| Path | Purpose |
|---|---|
| `common/generate_baseline.py` | In-memory Resample and stratified Bernoulli sampling (SBS). |
| `common/prepare_binary_matrix.py` | Converts a patient-level table to the binary matrix used by the neural generators. |
| `VAE/` | Tabular variational autoencoder training and Bernoulli sampling. |
| `medGAN/` | medGAN autoencoder, adversarial training, and sampling. |
| `CorGAN/` | CorGAN convolutional autoencoder, adversarial training, and sampling. |
| `HALO/` | Visit-level dataset construction, autoregressive model training, sampling, and patient-level aggregation. |
| `EHRDiff/` | EHRDiff training and deterministic EDM sampling. |

The patient-level input contains the four binary sensitive attributes `age80`, `male`, `white`, and `private`, followed by binary phenotype indicators. HALO additionally uses `subject_id` and `hadm_id` to represent ordered admissions.

## Dependencies

- Resample and SBS require NumPy and pandas.
- VAE, CorGAN, and HALO use Python 3.10 and `requirements-torch.txt`.
- medGAN uses Python 3.7, TensorFlow 1.x, and `medGAN/requirements.txt`.
- EHRDiff uses Python 3.9 and `EHRDiff/requirements.txt`. Its model, loss, EMA, and sampler modules are loaded from [sczzz3/EHRDiff](https://github.com/sczzz3/EHRDiff) at commit `26f064df711735af5d6c0d746dfd71ef2672c152`.

Licenses and citations for third-party methods are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
