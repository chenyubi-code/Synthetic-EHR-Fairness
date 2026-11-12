"""Small artificial-data training/sampling checks; never reads clinical files."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generation"))
from baselines import generate_resample, generate_sbs
from prepare_binary_matrix import prepare_binary_matrix
from matrix_to_csv import convert_matrix


def run(script, *args):
    result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                            capture_output=True, text=True, timeout=120, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f"{script} failed:\n{result.stdout}\n{result.stderr}")


def main():
    rng = np.random.default_rng(37)
    columns = ["age80", "male", "white", "private"] + [f"P{i}" for i in range(343)]
    artificial = pd.DataFrame(rng.binomial(1, .3, (32, 347)), columns=columns)
    first = generate_resample(artificial, 48, 17)
    pd.testing.assert_frame_equal(first, generate_resample(artificial, 48, 17))
    source_rows = set(map(tuple, artificial.to_numpy()))
    assert all(tuple(row) in source_rows for row in first.to_numpy())
    conditional = artificial.iloc[:, :6].copy()
    conditional["P0"] = conditional["age80"]
    conditional["P1"] = conditional["male"]
    sampled = generate_sbs(conditional, 100, 3)
    assert np.array_equal(sampled.P0, sampled.age80)
    assert np.array_equal(sampled.P1, sampled.male)
    matrix, names = prepare_binary_matrix(artificial, True)
    pd.testing.assert_frame_equal(convert_matrix(matrix, names), artificial.astype(np.int8))
    invalid = artificial.copy()
    invalid.iloc[0, 4] = np.nan
    try:
        prepare_binary_matrix(invalid)
        raise AssertionError("Missing binary values were not rejected")
    except ValueError:
        pass
    # Alternative definitions retain their explicit names throughout generation.
    for old, new in [("age80", "age85"), ("white", "black")]:
        alternate = conditional.rename(columns={old: new})
        labels = [new if name == old else name for name in columns[:4]]
        sampled_alt = generate_sbs(alternate, 100, 3, labels)
        pd.testing.assert_frame_equal(sampled_alt.rename(columns={new: old}), sampled)
        sampled_rows = generate_resample(alternate, 48, 17, labels)
        assert list(sampled_rows.columns) == list(alternate.columns)
        _, alternate_names = prepare_binary_matrix(alternate, sensitive_columns=labels)
        assert alternate_names == list(alternate.columns)
        try:
            prepare_binary_matrix(alternate)
            raise AssertionError("Alternative labels accepted without explicit schema")
        except ValueError:
            pass
    print("PASS: Resample/SBS, fixed seeds, conditional marginals, matrix schema/validation")

    import torch
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="ehr_artificial_test_") as temporary:
        folder = Path(temporary)
        input_csv = folder / "artificial.csv"
        artificial.to_csv(input_csv, index=False)
        run("generation/vae.py", "--input-csv", input_csv,
            "--output-csv", folder / "vae.csv", "--model-path", folder / "vae.pt",
            "--seed", 3, "--n-generate", 8, "--epochs", 1, "--batch-size", 16)
        vae_output = pd.read_csv(folder / "vae.csv")
        assert vae_output.shape == (8, 347) and np.isin(vae_output, [0, 1]).all()
        state = torch.load(folder / "vae.pt", map_location="cpu")
        assert state["input_dim"] == 347 and state["columns"] == columns
        alternate = artificial.rename(columns={"white": "black"})
        alternate_csv = folder / "alternate.csv"
        alternate.to_csv(alternate_csv, index=False)
        run("generation/vae.py", "--input-csv", alternate_csv,
            "--output-csv", folder / "alternate_vae.csv",
            "--seed", 3, "--n-generate", 8, "--epochs", 1, "--batch-size", 16,
            "--sensitive-columns", "age80", "male", "black", "private")
        assert list(pd.read_csv(folder / "alternate_vae.csv").columns) == list(alternate.columns)
        print("PASS: VAE one-epoch training, checkpoint, and binary sampling")

        np.save(folder / "artificial.npy", artificial.to_numpy(dtype=np.float32), allow_pickle=False)
        run("generation/corgan.py", "--DATASETPATH", folder / "artificial.npy",
            "--expPATH", folder / "corgan", "--manual_seed", 3,
            "--training", "true", "--generate", "true", "--n_epochs", 1,
            "--n_epochs_pretrain", 1, "--batch_size", 8, "--n_iter_D", 1,
            "--n_generate", 8, "--cuda", "false")
        assert (folder / "corgan/model_epoch_1.pth").is_file()
        probs = np.load(folder / "corgan/synthetic_prob.npy", allow_pickle=False)
        assert probs.shape == (8, 347) and np.isfinite(probs).all()
        assert convert_matrix(probs, columns, probabilities=True, seed=3).shape == (8, 347)
        print("PASS: CorGAN pretraining, adversarial update, final checkpoint and sampling")

        visits = artificial.iloc[:, :8].copy()
        visits["P0"] = 1
        visits.insert(0, "hadm_id", np.arange(1000, 1032))
        visits.insert(0, "subject_id", np.arange(32))
        visits.to_csv(folder / "visits.csv", index=False)
        visits.rename(columns={"age80": "age85"}).to_csv(folder / "alternate_visits.csv", index=False)
        run("generation/halo/build_halo_dataset_repeat.py", "--input_csv", folder / "alternate_visits.csv",
            "--out_dir", folder / "alternate_halo_data", "--label_cols", "age85", "male", "white", "private")
        import pickle
        with open(folder / "alternate_halo_data/idToLabel.pkl", "rb") as handle:
            assert pickle.load(handle) == ["age85", "male", "white", "private"]
        run("generation/halo/build_halo_dataset_repeat.py", "--input_csv", folder / "visits.csv",
            "--out_dir", folder / "halo_data")
        run("generation/halo/train_halo_repeat.py", "--data_dir", folder / "halo_data",
            "--save_dir", folder / "halo_model", "--seed", 3, "--epochs", 1,
            "--batch_size", 8, "--n_positions", 8, "--n_embd", 16, "--n_layer", 1, "--n_head", 2)
        run("generation/halo/generate_halo_repeat.py", "--data_dir", folder / "halo_data",
            "--model_path", folder / "halo_model/halo_model", "--out_pkl", folder / "halo.pkl",
            "--seed", 1003, "--n_generate", 8, "--sample_batch_size", 4)
        run("generation/halo/convert_halo_repeat.py", "--data_dir", folder / "halo_data",
            "--synthetic_pkl", folder / "halo.pkl", "--output_csv", folder / "halo.csv")
        halo_output = pd.read_csv(folder / "halo.csv")
        assert halo_output.shape == (8, 8) and np.isin(halo_output, [0, 1]).all()
        print("PASS: HALO preparation, training, checkpoint restoration, sampling and aggregation")
    print("Not exercised: TensorFlow 1.x medGAN and externally supplied EHRDiff modules.")


if __name__ == "__main__":
    main()
