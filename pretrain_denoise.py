import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from dataset import stratified_split_indices
from model import TinyIQTransformerBackbone, count_parameters
from pretrain import patchify_iq


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class IQDenoiseDataset(Dataset):
    """Load paired noisy/clean IQ data saved by data.py."""

    def __init__(self, npz_path: Path) -> None:
        if not npz_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=False)
        required_keys = {"x_noisy", "x_clean", "y", "snr"}
        missing = required_keys - set(data.files)
        if missing:
            raise KeyError(f"Denoising data is missing keys: {sorted(missing)}")

        self.x_noisy = data["x_noisy"].astype(np.float32)
        self.x_clean = data["x_clean"].astype(np.float32)
        self.y = data["y"].astype(np.int64)
        self.snr = data["snr"].astype(np.float32)

        assert self.x_noisy.shape == self.x_clean.shape
        assert self.x_noisy.ndim == 3
        assert self.x_noisy.shape[1] == 2
        assert self.y.shape == (self.x_noisy.shape[0],)
        assert self.snr.shape == (self.x_noisy.shape[0],)
        assert not np.isnan(self.x_noisy).any()
        assert not np.isnan(self.x_clean).any()
        assert not np.isinf(self.x_noisy).any()
        assert not np.isinf(self.x_clean).any()

    def __len__(self) -> int:
        return int(self.x_noisy.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy = torch.from_numpy(self.x_noisy[index])
        clean = torch.from_numpy(self.x_clean[index])
        assert noisy.shape == clean.shape
        assert noisy.ndim == 2 and noisy.shape[0] == 2
        return noisy, clean


class DenoisingIQPretrainer(nn.Module):
    """Tiny IQ Transformer trained to reconstruct clean IQ patches from noisy IQ."""

    def __init__(
        self,
        seq_len: int = 128,
        patch_size: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.backbone = TinyIQTransformerBackbone(
            seq_len=seq_len,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.patch_dim = 2 * patch_size
        self.recon_head = nn.Linear(hidden_dim, self.patch_dim)

    def forward(self, x_noisy: torch.Tensor) -> torch.Tensor:
        encoded_tokens = self.backbone(x_noisy, return_tokens=True)
        patch_tokens = encoded_tokens[:, 1:]
        pred_clean_patches = self.recon_head(patch_tokens)
        return pred_clean_patches


def reconstruction_loss(pred_patches: torch.Tensor, target_patches: torch.Tensor) -> torch.Tensor:
    assert pred_patches.shape == target_patches.shape
    return torch.mean((pred_patches - target_patches) ** 2)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    patch_size: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_noisy, x_clean in loader:
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)
        target_patches = patchify_iq(x_clean, patch_size=patch_size)

        optimizer.zero_grad(set_to_none=True)
        pred_patches = model(x_noisy)
        loss = reconstruction_loss(pred_patches, target_patches)
        loss.backward()
        optimizer.step()

        batch_size = x_noisy.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_reconstruction(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for x_noisy, x_clean in loader:
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)
        target_patches = patchify_iq(x_clean, patch_size=patch_size)
        pred_patches = model(x_noisy)
        loss = reconstruction_loss(pred_patches, target_patches)

        batch_size = x_noisy.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/iq_4mods_awgn_clean_n20000.npz"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/denoise_backbone.pt"))
    args = parser.parse_args()

    assert args.epochs > 0
    assert args.batch_size > 0
    assert args.lr > 0
    assert args.weight_decay >= 0

    set_seed(args.seed)
    device = torch.device("cpu")

    dataset = IQDenoiseDataset(args.data)
    train_idx, val_idx, _ = stratified_split_indices(
        labels=dataset.y,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=args.seed,
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = DenoisingIQPretrainer(
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    x_noisy, x_clean = next(iter(train_loader))
    x_noisy = x_noisy.to(device)
    x_clean = x_clean.to(device)
    target_patches = patchify_iq(x_clean, patch_size=args.patch_size)
    with torch.no_grad():
        pred_patches = model(x_noisy)
        init_loss = reconstruction_loss(pred_patches, target_patches)

    print("=== Denoising pretraining setup ===")
    print(f"Device: {device}")
    print(f"Data file: {args.data}")
    print(f"Train samples for SSL: {len(train_idx)}")
    print(f"Val samples for SSL: {len(val_idx)}")
    print(f"Noisy input shape: {tuple(x_noisy.shape)}")
    print(f"Clean target shape: {tuple(x_clean.shape)}")
    print(f"Target clean patches shape: {tuple(target_patches.shape)}")
    print(f"Predicted clean patches shape: {tuple(pred_patches.shape)}")
    print(f"Initial clean-target MSE on one batch: {float(init_loss.item()):.6f}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    assert pred_patches.shape == target_patches.shape
    assert torch.isfinite(pred_patches).all(), "Predicted patches contain NaN or Inf"

    print("\n=== SSL pretraining: noisy IQ -> clean IQ ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            patch_size=args.patch_size,
        )
        val_loss = evaluate_reconstruction(
            model=model,
            loader=val_loader,
            device=device,
            patch_size=args.patch_size,
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_clean_mse={train_loss:.6f} | "
            f"val_clean_mse={val_loss:.6f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "config": {
                "seq_len": args.seq_len,
                "patch_size": args.patch_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "objective": "noisy_to_clean_denoising",
            },
        },
        args.out,
    )
    print(f"\nSaved denoising pretrained backbone checkpoint: {args.out}")
    print("\nPASS: denoising pretraining sanity checks completed.")


if __name__ == "__main__":
    main()
