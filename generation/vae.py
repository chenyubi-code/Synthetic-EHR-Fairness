"""Tabular VAE training and sampling with the study configuration."""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def logit_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x) - np.log1p(-x)

class TabularVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        min_logvar: float = -8.0,
        max_logvar: float = 6.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.decoder = nn.Sequential(*decoder_layers)
        self.output_layer = nn.Linear(prev_dim, input_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def initialize_output_bias(self, prevalence: np.ndarray) -> None:
        bias = torch.from_numpy(logit_np(prevalence).astype(np.float32))
        with torch.no_grad():
            self.output_layer.bias.copy_(bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), self.min_logvar, self.max_logvar)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z)
        return self.output_layer(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar

def vae_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    pos_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_element = F.binary_cross_entropy_with_logits(
        logits,
        x,
        pos_weight=pos_weight,
        reduction="none",
    )
    recon = recon_element.sum(dim=1).mean()
    kld = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return recon + beta * kld, recon, kld


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--n-generate', type=int, required=True)
    parser.add_argument('--model-path', type=Path, help='Checkpoint output; defaults to output CSV with .pt suffix')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    set_seed(args.seed)
    if min(args.epochs, args.batch_size, args.n_generate) <= 0:
        parser.error('epochs, batch-size, and n-generate must be positive')
    try:
        from .prepare_binary_matrix import prepare_binary_matrix
    except ImportError:
        from prepare_binary_matrix import prepare_binary_matrix
    x, columns = prepare_binary_matrix(pd.read_csv(args.input_csv))
    prevalence = x.mean(axis=0).astype(np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=args.batch_size,
                        shuffle=True, drop_last=False, num_workers=0,
                        pin_memory=(device.type == 'cuda'))
    model = TabularVAE(x.shape[1], latent_dim=64, hidden_dims=(512, 256),
                       dropout=0.1, min_logvar=-8.0, max_logvar=6.0)
    model.initialize_output_bias(prevalence)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5)
    pos_weight = np.clip((1.0 - prevalence) / np.clip(prevalence, 1e-6, None), 1.0, 5.0).astype(np.float32)
    pos_weight = torch.from_numpy(pos_weight).to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        beta = 0.1 * min(1.0, epoch / 200)
        total_loss = 0.0
        total_examples = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model(batch_x)
            loss, _, _ = vae_loss(logits, batch_x, mu, logvar, beta, pos_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite VAE training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch_x.shape[0]
            total_examples += batch_x.shape[0]
        avg_loss = total_loss / total_examples
        scheduler.step(avg_loss)
        if epoch == 1 or epoch % 10 == 0:
            print('Epoch', epoch, 'loss', avg_loss, flush=True)

    model_path = args.model_path or args.output_csv.with_suffix('.pt')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'columns': columns, 'input_dim': x.shape[1],
                'latent_dim': 64, 'hidden_dims': (512, 256), 'dropout': 0.1,
                'seed': args.seed, 'epochs': args.epochs}, model_path)
    # Use a separate deterministic stream for generation.
    set_seed(args.seed + 1000)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, args.n_generate, 4096):
            size = min(4096, args.n_generate - start)
            latent = torch.randn(size, 64, device=device)
            probabilities.append(torch.sigmoid(model.decode_logits(latent)).cpu().numpy().astype(np.float32))
    probabilities = np.clip(np.vstack(probabilities), 1e-7, 1.0 - 1e-7)
    rng = np.random.default_rng(args.seed + 1000)
    binary = rng.binomial(1, probabilities).astype(np.int8)
    if not np.isfinite(probabilities).all() or probabilities.mean() < 1e-4 or binary.sum() == 0:
        raise RuntimeError('Degenerate VAE output; inspect training losses before proceeding.')
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(binary, columns=columns).to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
