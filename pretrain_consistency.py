import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import stratified_split_indices
from device_utils import print_device_report, resolve_device, seed_cuda
from model import TinyIQTransformerBackbone, count_parameters
from pretrain_noisy2noisy import IQNoisyPairDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_cuda(seed)


class ConsistencyIQPretrainer(nn.Module):
    """Tiny IQ Transformer trained with representation-level view consistency."""

    def __init__(
        self,
        seq_len: int = 128,
        patch_size: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        projector: str = "identity",
        projector_hidden_dim: int = 128,
        projector_out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = TinyIQTransformerBackbone(
            seq_len=seq_len,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        if projector_out_dim is None:
            projector_out_dim = hidden_dim
        if projector == "identity":
            self.projector = nn.Identity()
        elif projector == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, projector_hidden_dim),
                nn.GELU(),
                nn.Linear(projector_hidden_dim, projector_out_dim),
            )
        else:
            raise ValueError(f"Unknown projector: {projector}")

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_a = self.backbone(x_a)
        h_b = self.backbone(x_b)
        z_a = self.projector(h_a)
        z_b = self.projector(h_b)
        return z_a, z_b


def invariance_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    assert z_a.shape == z_b.shape
    cosine = torch.nn.functional.cosine_similarity(z_a, z_b, dim=1)
    return 1.0 - cosine.mean()


def variance_loss(z_a: torch.Tensor, z_b: torch.Tensor, target_std: float, eps: float) -> torch.Tensor:
    assert z_a.shape == z_b.shape
    std_a = torch.sqrt(z_a.var(dim=0, unbiased=False) + eps)
    std_b = torch.sqrt(z_b.var(dim=0, unbiased=False) + eps)
    loss_a = torch.relu(target_std - std_a).mean()
    loss_b = torch.relu(target_std - std_b).mean()
    return 0.5 * (loss_a + loss_b)


def consistency_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    variance_weight: float,
    variance_target: float,
    variance_eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    inv = invariance_loss(z_a, z_b)
    var = variance_loss(z_a, z_b, target_std=variance_target, eps=variance_eps)
    total = inv + variance_weight * var
    return total, {
        "inv_loss": float(inv.detach().item()),
        "var_loss": float(var.detach().item()),
        "total_loss": float(total.detach().item()),
    }


@torch.no_grad()
def representation_stats(z_a: torch.Tensor, z_b: torch.Tensor) -> dict[str, float]:
    z = torch.cat([z_a, z_b], dim=0)
    cosine = torch.nn.functional.cosine_similarity(z_a, z_b, dim=1)
    dim_std = z.std(dim=0, unbiased=False)
    return {
        "cosine": float(cosine.mean().item()),
        "z_std": float(z.std(unbiased=False).item()),
        "z_dim_std_mean": float(dim_std.mean().item()),
        "z_dim_std_min": float(dim_std.min().item()),
        "z_norm": float(z.norm(dim=1).mean().item()),
    }


def add_weighted_stats(total: dict[str, float], stats: dict[str, float], batch_size: int) -> None:
    for key, value in stats.items():
        total[key] = total.get(key, 0.0) + value * batch_size


def finalize_stats(total: dict[str, float], total_samples: int) -> dict[str, float]:
    return {key: value / total_samples for key, value in total.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    variance_weight: float,
    variance_target: float,
    variance_eps: float,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_stats: dict[str, float] = {}

    for x_a, x_b in loader:
        x_a = x_a.to(device)
        x_b = x_b.to(device)

        optimizer.zero_grad(set_to_none=True)
        z_a, z_b = model(x_a, x_b)
        loss, loss_stats = consistency_loss(
            z_a,
            z_b,
            variance_weight=variance_weight,
            variance_target=variance_target,
            variance_eps=variance_eps,
        )
        loss.backward()
        optimizer.step()

        batch_size = x_a.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        batch_stats = representation_stats(z_a.detach(), z_b.detach())
        batch_stats.update(loss_stats)
        add_weighted_stats(total_stats, batch_stats, batch_size)

    return total_loss / total_samples, finalize_stats(total_stats, total_samples)


@torch.no_grad()
def evaluate_consistency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    variance_weight: float,
    variance_target: float,
    variance_eps: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_stats: dict[str, float] = {}

    for x_a, x_b in loader:
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        z_a, z_b = model(x_a, x_b)
        loss, loss_stats = consistency_loss(
            z_a,
            z_b,
            variance_weight=variance_weight,
            variance_target=variance_target,
            variance_eps=variance_eps,
        )

        batch_size = x_a.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        batch_stats = representation_stats(z_a, z_b)
        batch_stats.update(loss_stats)
        add_weighted_stats(total_stats, batch_stats, batch_size)

    return total_loss / total_samples, finalize_stats(total_stats, total_samples)


def collapse_note(stats: dict[str, float], std_threshold: float) -> str:
    if stats["z_dim_std_mean"] < std_threshold or stats["z_std"] < std_threshold:
        return " | COLLAPSE_WARN"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/iq_4mods_awgn_views_n20000.npz"))
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
    parser.add_argument("--out", type=Path, default=Path("checkpoints/consistency_backbone.pt"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--collapse-std-threshold", type=float, default=1e-3)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--variance-target", type=float, default=0.2)
    parser.add_argument("--variance-eps", type=float, default=1e-4)
    parser.add_argument("--projector", choices=["identity", "mlp"], default="identity")
    parser.add_argument("--projector-hidden-dim", type=int, default=128)
    parser.add_argument("--projector-out-dim", type=int, default=None)
    args = parser.parse_args()

    assert args.epochs > 0
    assert args.batch_size > 0
    assert args.lr > 0
    assert args.weight_decay >= 0
    assert args.collapse_std_threshold > 0
    assert args.variance_weight >= 0
    assert args.variance_target > 0
    assert args.variance_eps > 0
    assert args.projector_hidden_dim > 0
    if args.projector_out_dim is not None:
        assert args.projector_out_dim > 0

    set_seed(args.seed)
    device = resolve_device(args.device)

    dataset = IQNoisyPairDataset(args.data)
    train_idx, val_idx, _ = stratified_split_indices(
        labels=dataset.y,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=args.seed,
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            "Train/val split is empty. Use a larger dataset for consistency pretraining."
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

    model = ConsistencyIQPretrainer(
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        projector=args.projector,
        projector_hidden_dim=args.projector_hidden_dim,
        projector_out_dim=args.projector_out_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    x_a, x_b = next(iter(train_loader))
    x_a = x_a.to(device)
    x_b = x_b.to(device)
    with torch.no_grad():
        z_a, z_b = model(x_a, x_b)
        init_loss, init_loss_stats = consistency_loss(
            z_a,
            z_b,
            variance_weight=args.variance_weight,
            variance_target=args.variance_target,
            variance_eps=args.variance_eps,
        )
        init_stats = representation_stats(z_a, z_b)
        init_stats.update(init_loss_stats)

    print("=== Representation consistency pretraining setup ===")
    print_device_report(device)
    print(f"Data file: {args.data}")
    print(f"Train samples for SSL: {len(train_idx)}")
    print(f"Val samples for SSL: {len(val_idx)}")
    print(f"View A input shape: {tuple(x_a.shape)}")
    print(f"View B input shape: {tuple(x_b.shape)}")
    print(f"View A device: {x_a.device}")
    print(f"View B device: {x_b.device}")
    print(f"z_a shape: {tuple(z_a.shape)}")
    print(f"z_b shape: {tuple(z_b.shape)}")
    print(f"Projector: {args.projector}")
    if args.projector == "mlp":
        print(f"Projector hidden dim: {args.projector_hidden_dim}")
        print(f"Projector out dim: {z_a.shape[1]}")
    print(f"Variance weight: {args.variance_weight}")
    print(f"Variance target std: {args.variance_target}")
    print(f"Initial consistency loss: {float(init_loss.item()):.6f}")
    print(
        "Initial stats | "
        f"inv_loss={init_stats['inv_loss']:.6f} | "
        f"var_loss={init_stats['var_loss']:.6f} | "
        f"cos={init_stats['cosine']:.4f} | "
        f"z_std={init_stats['z_std']:.6f} | "
        f"dim_std_mean={init_stats['z_dim_std_mean']:.6f} | "
        f"dim_std_min={init_stats['z_dim_std_min']:.6f} | "
        f"z_norm={init_stats['z_norm']:.4f}"
        f"{collapse_note(init_stats, args.collapse_std_threshold)}"
    )
    print(f"Trainable parameters: {count_parameters(model):,}")

    assert z_a.shape == z_b.shape
    assert z_a.ndim == 2
    assert torch.isfinite(z_a).all(), "z_a contains NaN or Inf"
    assert torch.isfinite(z_b).all(), "z_b contains NaN or Inf"

    print("\n=== SSL pretraining: representation consistency ===")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            variance_weight=args.variance_weight,
            variance_target=args.variance_target,
            variance_eps=args.variance_eps,
        )
        val_loss, val_stats = evaluate_consistency(
            model=model,
            loader=val_loader,
            device=device,
            variance_weight=args.variance_weight,
            variance_target=args.variance_target,
            variance_eps=args.variance_eps,
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"train_inv={train_stats['inv_loss']:.6f} | "
            f"train_var={train_stats['var_loss']:.6f} | "
            f"train_cos={train_stats['cosine']:.4f} | "
            f"train_z_std={train_stats['z_std']:.6f} | "
            f"train_dim_std={train_stats['z_dim_std_mean']:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_inv={val_stats['inv_loss']:.6f} | "
            f"val_var={val_stats['var_loss']:.6f} | "
            f"val_cos={val_stats['cosine']:.4f} | "
            f"val_z_std={val_stats['z_std']:.6f} | "
            f"val_dim_std={val_stats['z_dim_std_mean']:.6f} | "
            f"val_dim_min={val_stats['z_dim_std_min']:.6f} | "
            f"val_z_norm={val_stats['z_norm']:.4f}"
            f"{collapse_note(val_stats, args.collapse_std_threshold)}",
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
                "objective": "representation_consistency",
                "loss": "1_minus_cosine_similarity_plus_variance_regularization",
                "projector": args.projector,
                "projector_hidden_dim": args.projector_hidden_dim,
                "projector_out_dim": z_a.shape[1],
                "variance_weight": args.variance_weight,
                "variance_target": args.variance_target,
                "variance_eps": args.variance_eps,
            },
        },
        args.out,
    )
    print(f"\nSaved consistency pretrained backbone checkpoint: {args.out}")
    print("\nPASS: representation consistency pretraining sanity checks completed.")


if __name__ == "__main__":
    main()
