#!/usr/bin/env python
import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train HALO for one repeat seed.")
    parser.add_argument("--methods_halo_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_positions", type=int, default=48)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_config(args, data_dir):
    sys.path.insert(0, str(Path(args.methods_halo_dir).resolve()))
    from config import HALOConfig

    index_to_code = pickle.load(open(Path(data_dir) / "indexToCode.pkl", "rb"))
    labels = pickle.load(open(Path(data_dir) / "idToLabel.pkl", "rb"))
    code_vocab_size = len(index_to_code)
    label_vocab_size = len(labels)
    special_vocab_size = 3
    return HALOConfig(
        total_vocab_size=code_vocab_size + label_vocab_size + special_vocab_size,
        code_vocab_size=code_vocab_size,
        label_vocab_size=label_vocab_size,
        special_vocab_size=special_vocab_size,
        n_positions=args.n_positions,
        n_ctx=args.n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        batch_size=args.batch_size,
        epoch=args.epochs,
        lr=args.lr,
    )


def get_batch(dataset, loc, batch_size, config):
    ehr = dataset[loc : loc + batch_size]
    batch_ehr = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
    batch_mask = np.zeros((len(ehr), config.n_ctx, 1))
    for i, patient in enumerate(ehr):
        visits = patient["visits"][: config.n_ctx - 2]
        for j, visit in enumerate(visits):
            batch_ehr[i, j + 2][visit] = 1
            batch_mask[i, j + 2] = 1
        batch_ehr[i, 1, config.code_vocab_size : config.code_vocab_size + config.label_vocab_size] = np.array(
            patient["labels"]
        )
        batch_ehr[i, len(visits) + 1, config.code_vocab_size + config.label_vocab_size + 1] = 1
        batch_ehr[i, len(visits) + 2 :, config.code_vocab_size + config.label_vocab_size + 2] = 1
    batch_mask[:, 1] = 1
    batch_ehr[:, 0, config.code_vocab_size + config.label_vocab_size] = 1
    return batch_ehr, batch_mask[:, 1:, :]


def main():
    args = parse_args()
    set_seed(args.seed)
    sys.path.insert(0, str(Path(args.methods_halo_dir).resolve()))
    from model import HALOModel

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "halo_model"

    config = make_config(args, data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = pickle.load(open(data_dir / "trainDataset.pkl", "rb"))
    val_dataset = pickle.load(open(data_dir / "valDataset.pkl", "rb"))

    model = HALOModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_val_loss = float("inf")

    for epoch in tqdm(range(config.epoch)):
        np.random.shuffle(train_dataset)
        train_losses = []
        for i in range(0, len(train_dataset), config.batch_size):
            model.train()
            batch_ehr, batch_mask = get_batch(train_dataset, i, config.batch_size, config)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            loss, _, _ = model(
                batch_ehr,
                position_ids=None,
                ehr_labels=batch_ehr,
                ehr_masks=batch_mask,
                pos_loss_weight=config.pos_loss_weight,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(val_dataset), config.batch_size):
                batch_ehr, batch_mask = get_batch(val_dataset, i, config.batch_size, config)
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
                val_loss, _, _ = model(
                    batch_ehr,
                    position_ids=None,
                    ehr_labels=batch_ehr,
                    ehr_masks=batch_mask,
                    pos_loss_weight=config.pos_loss_weight,
                )
                val_losses.append(val_loss.item())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        print(f"Epoch {epoch} | train_loss={avg_train:.7f} | val_loss={avg_val:.7f}", flush=True)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_loss": avg_val,
                    "seed": args.seed,
                    "config": vars(config),
                },
                model_path,
            )
            print("Saved best model:", model_path, flush=True)


if __name__ == "__main__":
    main()
