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
    parser = argparse.ArgumentParser(description="Generate HALO synthetic patients.")
    parser.add_argument("--methods_halo_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_pkl", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_generate", type=int, default=30000)
    parser.add_argument("--sample_batch_size", type=int, default=256)
    parser.add_argument("--n_positions", type=int, default=48)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_config(args):
    sys.path.insert(0, str(Path(args.methods_halo_dir).resolve()))
    from config import HALOConfig

    data_dir = Path(args.data_dir)
    index_to_code = pickle.load(open(data_dir / "indexToCode.pkl", "rb"))
    labels = pickle.load(open(data_dir / "idToLabel.pkl", "rb"))
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
        sample_batch_size=args.sample_batch_size,
    )


def sample_sequence(model, config, length, context, batch_size, device, sample=True):
    empty = torch.zeros((1, 1, config.total_vocab_size), device=device, dtype=torch.float32).repeat(batch_size, 1, 1)
    context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    prev = context.unsqueeze(1)
    with torch.no_grad():
        for _ in range(length - 1):
            prev = model.sample(torch.cat((prev, empty), dim=1), sample)
            end_token = config.code_vocab_size + config.label_vocab_size + 1
            if torch.sum(torch.sum(prev[:, :, end_token], dim=1).bool().int(), dim=0).item() == batch_size:
                break
    return prev.cpu().detach().numpy()


def convert_ehr(config, ehrs):
    outputs = []
    for ehr in ehrs:
        labels = ehr[1][config.code_vocab_size : config.code_vocab_size + config.label_vocab_size]
        visits = []
        for j in range(2, len(ehr)):
            visit = ehr[j]
            visit_codes = []
            end = False
            for idx in np.nonzero(visit)[0]:
                if idx < config.code_vocab_size:
                    visit_codes.append(int(idx))
                elif idx == config.code_vocab_size + config.label_vocab_size + 1:
                    end = True
            if visit_codes:
                visits.append(visit_codes)
            if end:
                break
        outputs.append({"visits": visits, "labels": labels})
    return outputs


def main():
    args = parse_args()
    set_seed(args.seed)
    sys.path.insert(0, str(Path(args.methods_halo_dir).resolve()))
    from model import HALOModel

    config = make_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HALOModel(config).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    stoken = np.zeros(config.total_vocab_size)
    stoken[config.code_vocab_size + config.label_vocab_size] = 1

    synthetic = []
    for i in tqdm(range(0, args.n_generate, config.sample_batch_size)):
        batch_size = min(args.n_generate - i, config.sample_batch_size)
        batch = sample_sequence(model, config, config.n_ctx, stoken, batch_size, device, sample=True)
        synthetic.extend(convert_ehr(config, batch))

    synthetic = synthetic[: args.n_generate]
    out_pkl = Path(args.out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(synthetic, open(out_pkl, "wb"))
    print("Saved:", out_pkl)
    print("Synthetic subjects:", len(synthetic))


if __name__ == "__main__":
    main()
