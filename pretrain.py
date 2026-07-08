import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import IQDataset, stratified_split_indices
from model import TinyIQTransformerBackbone, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def patchify_iq(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert IQ sequence [B, 2, L] into raw patches [B, P, 2 * patch_size]."""
    assert x.ndim == 3, f"Expected x [B, 2, L], got {tuple(x.shape)}"
    assert x.shape[1] == 2, f"Expected IQ channel dim=2, got {tuple(x.shape)}"
    assert x.shape[2] % patch_size == 0, "Sequence length must be divisible by patch_size"

    batch_size = x.shape[0]
    num_patches = x.shape[2] // patch_size
    patch_dim = 2 * patch_size

    patches = x.reshape(batch_size, 2, num_patches, patch_size)
    patches = patches.permute(0, 2, 1, 3)
    patches = patches.reshape(batch_size, num_patches, patch_dim)
    return patches


def random_patch_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Return bool mask [B, P], where True means this patch is hidden."""
    assert 0.0 < mask_ratio < 1.0
    num_masked = max(1, int(round(num_patches * mask_ratio)))
    assert num_masked < num_patches, "At least one patch should remain visible"

    random_scores = torch.rand(batch_size, num_patches, device=device)
    mask_indices = torch.argsort(random_scores, dim=1)[:, :num_masked]
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    mask.scatter_(dim=1, index=mask_indices, value=True)

    assert mask.shape == (batch_size, num_patches)
    assert torch.all(mask.sum(dim=1) == num_masked)
    return mask


class MaskedIQPretrainer(nn.Module):
    """Tiny IQ Transformer trained to reconstruct masked raw IQ patches."""

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
        self.num_patches = self.backbone.num_patches
        self.patch_dim = 2 * patch_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.recon_head = nn.Linear(hidden_dim, self.patch_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert mask.ndim == 2, f"Expected mask [B, P], got {tuple(mask.shape)}"

        patch_tokens = self.backbone.patch_embed(x)
        assert mask.shape == patch_tokens.shape[:2], (
            f"Mask shape {tuple(mask.shape)} must match patch token shape {tuple(patch_tokens.shape[:2])}"
        )

        mask_token = self.mask_token.expand(x.shape[0], self.num_patches, -1)
        patch_tokens = torch.where(mask.unsqueeze(-1), mask_token, patch_tokens)

        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        tokens = tokens + self.backbone.pos_embed

        encoded = self.backbone.encoder(tokens)
        encoded = self.backbone.norm(encoded)

        patch_encoded = encoded[:, 1:]
        pred_patches = self.recon_head(patch_encoded)
        return pred_patches


def masked_reconstruction_loss(
    pred_patches: torch.Tensor,
    target_patches: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    assert pred_patches.shape == target_patches.shape
    assert mask.shape == pred_patches.shape[:2]

    pred_masked = pred_patches[mask]
    target_masked = target_patches[mask]

    assert pred_masked.ndim == 2
    assert pred_masked.shape[0] > 0
    return torch.mean((pred_masked - target_masked) ** 2)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float,
    patch_size: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, _ in loader:
        x = x.to(device)
        target_patches = patchify_iq(x, patch_size=patch_size)
        mask = random_patch_mask(
            batch_size=x.shape[0],
            num_patches=target_patches.shape[1],
            mask_ratio=mask_ratio,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        pred_patches = model(x, mask)
        loss = masked_reconstruction_loss(pred_patches, target_patches, mask)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_reconstruction(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
    patch_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for x, _ in loader:
        x = x.to(device)
        target_patches = patchify_iq(x, patch_size=patch_size)
        mask = random_patch_mask(
            batch_size=x.shape[0],
            num_patches=target_patches.shape[1],
            mask_ratio=mask_ratio,
            device=device,
        )
        pred_patches = model(x, mask)
        loss = masked_reconstruction_loss(pred_patches, target_patches, mask)

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/iq_4mods_awgn.npz"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/ssl_backbone.pt"))
    args = parser.parse_args()

    assert args.epochs > 0
    assert args.batch_size > 0
    assert args.lr > 0
    assert args.weight_decay >= 0
    assert 0.0 < args.mask_ratio < 1.0

    set_seed(args.seed)
    device = torch.device("cpu")

    dataset = IQDataset(args.data)
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

    model = MaskedIQPretrainer(
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

    x_batch, _ = next(iter(train_loader))
    x_batch = x_batch.to(device)
    target_patches = patchify_iq(x_batch, patch_size=args.patch_size)
    mask = random_patch_mask(
        batch_size=x_batch.shape[0],
        num_patches=target_patches.shape[1],
        mask_ratio=args.mask_ratio,
        device=device,
    )
    with torch.no_grad():
        pred_patches = model(x_batch, mask)
        init_loss = masked_reconstruction_loss(pred_patches, target_patches, mask)

    print("=== Masked reconstruction setup ===")
    print(f"Device: {device}")
    print(f"Data file: {args.data}")
    print(f"Train samples for SSL: {len(train_idx)}")
    print(f"Val samples for SSL: {len(val_idx)}")
    print(f"Input x shape: {tuple(x_batch.shape)}")
    print(f"Target raw patches shape: {tuple(target_patches.shape)}")
    print(f"Mask shape: {tuple(mask.shape)}")
    print(f"Masked patches per sample: {int(mask[0].sum().item())}/{mask.shape[1]}")
    print(f"Predicted patches shape: {tuple(pred_patches.shape)}")
    print(f"Initial masked MSE on one batch: {float(init_loss.item()):.6f}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    assert target_patches.shape == pred_patches.shape
    assert mask.shape == target_patches.shape[:2]
    assert torch.isfinite(pred_patches).all(), "Predicted patches contain NaN or Inf"

    print("\n=== SSL pretraining: masked patch reconstruction ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mask_ratio=args.mask_ratio,
            patch_size=args.patch_size,
        )
        val_loss = evaluate_reconstruction(
            model=model,
            loader=val_loader,
            device=device,
            mask_ratio=args.mask_ratio,
            patch_size=args.patch_size,
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_masked_mse={train_loss:.6f} | "
            f"val_masked_mse={val_loss:.6f}"
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
                "mask_ratio": args.mask_ratio,
            },
        },
        args.out,
    )
    print(f"\nSaved pretrained backbone checkpoint: {args.out}")
    print("\nPASS: masked reconstruction pretraining sanity checks completed.")


if __name__ == "__main__":
    main()
