import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import IQDataset, stratified_split_indices
from finetune import choose_labeled_subset, evaluate, train_one_epoch as train_classifier_one_epoch
from model import TinyIQClassifier
from pretrain import (
    MaskedIQPretrainer,
    evaluate_reconstruction,
    train_one_epoch as train_ssl_one_epoch,
)


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(dataset, indices, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def pretrain_backbone_for_seed(
    dataset: IQDataset,
    train_idx: list[int],
    val_idx: list[int],
    seed: int,
    args,
    device: torch.device,
) -> tuple[Path, float, float]:
    set_seed(seed)
    train_loader = make_loader(dataset, train_idx, args.ssl_batch_size, shuffle=True)
    val_loader = make_loader(dataset, val_idx, args.ssl_batch_size, shuffle=False)

    model = MaskedIQPretrainer(
        seq_len=128,
        patch_size=args.patch_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.ssl_lr,
        weight_decay=args.weight_decay,
    )

    train_loss = float("nan")
    val_loss = float("nan")
    for epoch in range(1, args.ssl_epochs + 1):
        train_loss = train_ssl_one_epoch(
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
        if not args.quiet:
            print(
                f"  SSL epoch {epoch:02d}/{args.ssl_epochs} | "
                f"train_masked_mse={train_loss:.6f} | "
                f"val_masked_mse={val_loss:.6f}",
                flush=True,
            )

    ckpt_path = args.checkpoint_dir / f"ssl_backbone_seed{seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "config": {
                "seed": seed,
                "patch_size": args.patch_size,
                "mask_ratio": args.mask_ratio,
                "ssl_epochs": args.ssl_epochs,
            },
        },
        ckpt_path,
    )
    return ckpt_path, train_loss, val_loss


def finetune_for_seed(
    dataset: IQDataset,
    labeled_train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    seed: int,
    args,
    device: torch.device,
    pretrained_path: Path | None,
) -> dict:
    set_seed(seed)
    train_loader = make_loader(dataset, labeled_train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(dataset, val_idx, args.batch_size, shuffle=False)
    test_loader = make_loader(dataset, test_idx, args.batch_size, shuffle=False)

    model = TinyIQClassifier().to(device)
    method = "ssl_pretrained" if pretrained_path is not None else "scratch"
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(checkpoint["backbone"], strict=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loss = float("nan")
    val_loss = float("nan")
    val_acc = float("nan")
    val_f1 = float("nan")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_classifier_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, num_classes=4)
        if not args.quiet:
            print(
                f"  {method} epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"val_macro_f1={val_f1:.4f}",
                flush=True,
            )

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device, num_classes=4)
    return {
        "seed": seed,
        "method": method,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
    }


def print_summary(rows: list[dict]) -> None:
    print("\n=== Multi-seed summary ===")
    for method in ["scratch", "ssl_pretrained"]:
        method_rows = [row for row in rows if row["method"] == method]
        acc = np.array([row["test_acc"] for row in method_rows], dtype=np.float64)
        f1 = np.array([row["test_macro_f1"] for row in method_rows], dtype=np.float64)
        loss = np.array([row["test_loss"] for row in method_rows], dtype=np.float64)

        print(
            f"{method:<15} | "
            f"test_acc={acc.mean():.4f} +/- {acc.std(ddof=0):.4f} | "
            f"test_macro_f1={f1.mean():.4f} +/- {f1.std(ddof=0):.4f} | "
            f"test_loss={loss.mean():.4f} +/- {loss.std(ddof=0):.4f}"
        )

    scratch = [row for row in rows if row["method"] == "scratch"]
    ssl = [row for row in rows if row["method"] == "ssl_pretrained"]
    scratch_by_seed = {row["seed"]: row for row in scratch}
    ssl_by_seed = {row["seed"]: row for row in ssl}

    print("\n=== Per-seed delta: SSL - Scratch ===")
    for seed in sorted(scratch_by_seed):
        acc_delta = ssl_by_seed[seed]["test_acc"] - scratch_by_seed[seed]["test_acc"]
        f1_delta = ssl_by_seed[seed]["test_macro_f1"] - scratch_by_seed[seed]["test_macro_f1"]
        print(f"seed={seed} | delta_acc={acc_delta:+.4f} | delta_macro_f1={f1_delta:+.4f}")


def save_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "method",
        "train_loss",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "test_loss",
        "test_acc",
        "test_macro_f1",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/iq_4mods_awgn.npz"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--label-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ssl-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ssl-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ssl-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/multiseed"))
    parser.add_argument("--out", type=Path, default=Path("results/multiseed_results.csv"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    original_stdout = sys.stdout
    log_handle = None
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_file.open("w", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.stdout, log_handle)

    assert len(args.seeds) > 0
    assert 0.0 < args.label_ratio <= 1.0
    assert args.epochs > 0
    assert args.ssl_epochs > 0
    assert 0.0 < args.mask_ratio < 1.0

    device = torch.device("cpu")
    dataset = IQDataset(args.data)
    rows = []

    print("=== Multi-seed reliability run ===")
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Label ratio: {args.label_ratio}")
    print(f"Finetune epochs: {args.epochs}")
    print(f"SSL pretrain epochs per seed: {args.ssl_epochs}")

    for seed in args.seeds:
        train_idx, val_idx, test_idx = stratified_split_indices(
            labels=dataset.y,
            train_ratio=0.7,
            val_ratio=0.15,
            seed=seed,
        )
        labeled_train_idx = choose_labeled_subset(
            labels=dataset.y,
            candidate_indices=train_idx,
            label_ratio=args.label_ratio,
            seed=seed,
        )

        print(f"\n--- Seed {seed} ---")
        print(f"Labeled train samples: {len(labeled_train_idx)}")

        ckpt_path, ssl_train_loss, ssl_val_loss = pretrain_backbone_for_seed(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            seed=seed,
            args=args,
            device=device,
        )
        print(
            f"SSL pretrain done | train_masked_mse={ssl_train_loss:.6f} | "
            f"val_masked_mse={ssl_val_loss:.6f}"
        )

        scratch_row = finetune_for_seed(
            dataset=dataset,
            labeled_train_idx=labeled_train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=seed,
            args=args,
            device=device,
            pretrained_path=None,
        )
        rows.append(scratch_row)
        print(
            f"Scratch test | acc={scratch_row['test_acc']:.4f} | "
            f"macro_f1={scratch_row['test_macro_f1']:.4f}"
        )

        ssl_row = finetune_for_seed(
            dataset=dataset,
            labeled_train_idx=labeled_train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=seed,
            args=args,
            device=device,
            pretrained_path=ckpt_path,
        )
        rows.append(ssl_row)
        print(
            f"SSL test     | acc={ssl_row['test_acc']:.4f} | "
            f"macro_f1={ssl_row['test_macro_f1']:.4f}"
        )

    print_summary(rows)
    save_csv(rows, args.out)
    print(f"\nSaved results CSV: {args.out}")
    if args.log_file is not None:
        print(f"Saved live log: {args.log_file}")
    print("\nPASS: multi-seed reliability check completed.")
    if log_handle is not None:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main()
