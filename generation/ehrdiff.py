#!/usr/bin/env python3
"""Train and sample EHRDiff with the study configuration."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-npy', type=Path, required=True)
    parser.add_argument('--mode', choices=('train', 'generate'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--n-generate', type=int, required=True)
    parser.add_argument('--port', type=int, default=61201)
    args = parser.parse_args()
    if not 1 <= args.n_generate <= 30000:
        parser.error('--n-generate must be between 1 and 30000 for the study sampler')
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from model.linear_model import LinearModel
    from model.ema import ExponentialMovingAverage
    from denoiser import EDMDenoiser
    from score_losses import EDMLoss
    from samplers import ablation_sampler
    from utils.util import set_seeds
    from transformers import get_cosine_schedule_with_warmup

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(args.port)
    dist.init_process_group(backend='gloo', rank=0, world_size=1)
    try:
        seed = args.seed if args.mode == 'train' else args.seed + 1000
        set_seeds(0, seed)
        model = EDMDenoiser(LinearModel(z_dim=348, time_dim=384,
            unit_dims=[1024, 384, 384, 384, 1024], use_cfg=False).to('cpu'),
            sigma_data=0.14, sigma_min=0.02, sigma_max=80.0)
        model = DistributedDataParallel(model.to('cpu'))

        def sample(latent):
            return ablation_sampler(latent, None, model, solver='heun', discretization='edm',
                stochastic=False, num_steps=32, sigma_min=0.02, sigma_max=80.0,
                rho=7.0, guid_scale=None)

        if args.mode == 'train':
            ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0)
            matrix = np.load(args.matrix, allow_pickle=False)
            if matrix.ndim != 2 or matrix.shape[1] != 348:
                raise ValueError('EHRDiff benchmark expects 347 binary features plus one all-zero padding column.')
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000,
                num_training_steps=(matrix.shape[0] // 1000 + 1) * 1000)
            loader = torch.utils.data.DataLoader(matrix, shuffle=True, batch_size=1000)
            loss_fn = EDMLoss(p_mean=-1.2, p_std=1.2, sigma_data=0.14, n_classes=None).get_loss
            step = 0
            for epoch in range(1000):
                for batch in loader:
                    x = batch.to('cpu').to(torch.float32)
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.mean(loss_fn(model, x, None))
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if step >= 1000 and step % 1000 == 0:
                        # Advance the sampler at each reporting interval before updating EMA.
                        model.eval()
                        with torch.no_grad():
                            ema.store(model.parameters())
                            ema.copy_to(model.parameters())
                            sample(torch.randn(matrix.shape, device='cpu'))
                            ema.restore(model.parameters())
                        model.train()
                        print('Epoch', epoch + 1, 'step', step + 1, 'loss', float(loss), flush=True)
                    step += 1
                    ema.update(model.parameters())
            args.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(),
                optimizer=optimizer.state_dict(), step=step), args.model_path)
        else:
            try:
                state = torch.load(args.model_path, map_location='cpu', weights_only=False)
            except TypeError:
                state = torch.load(args.model_path, map_location='cpu')
            model.load_state_dict(state['model'], strict=True)
            ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
            ema.load_state_dict(state['ema'])
            ema.copy_to(model.parameters())
            model.eval()
            # Generate one 30,000-row batch and keep the requested number of rows.
            with torch.no_grad():
                sampled = sample(torch.randn((30000, 348), device='cpu')).cpu().numpy()
            binary = np.rint(np.clip(sampled[:args.n_generate], 0, 1))
            args.output_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.output_npy, binary, allow_pickle=False)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
