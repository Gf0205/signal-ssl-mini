import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from compare_denoise_seeds import Tee
from dataset import IQDataset, stratified_split_indices
from finetune import choose_labeled_subset, evaluate, train_one_epoch as train_classifier_one_epoch
from model import TinyIQClassifier
from pretrain_denoise import (
    DenoisingIQPretrainer,
    IQDenoiseDataset,
    evaluate_reconstruction as evaluate_clean_reconstruction,
    train_one_epoch as train_clean_one_epoch,
)
from pretrain_noisy2noisy import (
    IQNoisyPairDataset,
    NoisyToNoisyIQPretrainer,
    evaluate_reconstruction as evaluate_noisy_reconstruction,
    train_one_epoch as train_noisy_one_epoch,
)


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


def pretrain_or_load_clean(
    dataset: IQDenoiseDataset,
    train_idx: list[int],
    val_idx: list[int],
    seed: int,
    args,
    device: torch.device,
) -> Path:
    ckpt_path = args.clean_checkpoint_dir / f"clean_backbone_seed{seed}.pt"
    if args.reuse_checkpoints and ckpt_path.exists():
        print(f"  Reusing NoisyA->Clean checkpoint: {ckpt_path}", flush=True)
        return ckpt_path

    set_seed(seed)
    train_loader = make_loader(dataset, train_idx, args.ssl_batch_size, shuffle=True)
    val_loader = make_loader(dataset, val_idx, args.ssl_batch_size, shuffle=False)
    model = DenoisingIQPretrainer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ssl_lr, weight_decay=args.weight_decay)

    train_loss = float("nan")
    val_loss = float("nan")
    for epoch in range(1, args.ssl_epochs + 1):
        train_loss = train_clean_one_epoch(model, train_loader, optimizer, device, args.patch_size)
        val_loss = evaluate_clean_reconstruction(model, val_loader, device, args.patch_size)
        if not args.quiet:
            print(
                f"  clean epoch {epoch:02d}/{args.ssl_epochs} | "
                f"train_clean_mse={train_loss:.6f} | val_clean_mse={val_loss:.6f}",
                flush=True,
            )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "config": {
                "seed": seed,
                "objective": "noisy_a_to_clean",
                "ssl_epochs": args.ssl_epochs,
            },
        },
        ckpt_path,
    )
    print(
        f"  NoisyA->Clean pretrain done | train_clean_mse={train_loss:.6f} | "
        f"val_clean_mse={val_loss:.6f}",
        flush=True,
    )
    return ckpt_path


def pretrain_or_load_noisy(
    dataset: IQNoisyPairDataset,
    train_idx: list[int],
    val_idx: list[int],
    seed: int,
    args,
    device: torch.device,
) -> Path:
    ckpt_path = args.noisy_checkpoint_dir / f"noisy2noisy_backbone_seed{seed}.pt"
    if args.reuse_checkpoints and ckpt_path.exists():
        print(f"  Reusing NoisyA->NoisyB checkpoint: {ckpt_path}", flush=True)
        return ckpt_path

    set_seed(seed)
    train_loader = make_loader(dataset, train_idx, args.ssl_batch_size, shuffle=True)
    val_loader = make_loader(dataset, val_idx, args.ssl_batch_size, shuffle=False)
    model = NoisyToNoisyIQPretrainer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ssl_lr, weight_decay=args.weight_decay)

    train_loss = float("nan")
    val_loss = float("nan")
    for epoch in range(1, args.ssl_epochs + 1):
        train_loss = train_noisy_one_epoch(model, train_loader, optimizer, device, args.patch_size)
        val_loss = evaluate_noisy_reconstruction(model, val_loader, device, args.patch_size)
        if not args.quiet:
            print(
                f"  noisy2noisy epoch {epoch:02d}/{args.ssl_epochs} | "
                f"train_noisy_mse={train_loss:.6f} | val_noisy_mse={val_loss:.6f}",
                flush=True,
            )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "config": {
                "seed": seed,
                "objective": "noisy_a_to_noisy_b",
                "ssl_epochs": args.ssl_epochs,
            },
        },
        ckpt_path,
    )
    print(
        f"  NoisyA->NoisyB pretrain done | train_noisy_mse={train_loss:.6f} | "
        f"val_noisy_mse={val_loss:.6f}",
        flush=True,
    )
    return ckpt_path


def load_backbone(model: TinyIQClassifier, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert "backbone" in checkpoint
    model.backbone.load_state_dict(checkpoint["backbone"], strict=True)


def finetune_for_method(
    dataset: IQDataset,
    labeled_train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    seed: int,
    method: str,
    checkpoint_path: Path | None,
    args,
    device: torch.device,
) -> dict:
    set_seed(seed)
    train_loader = make_loader(dataset, labeled_train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(dataset, val_idx, args.batch_size, shuffle=False)
    test_loader = make_loader(dataset, test_idx, args.batch_size, shuffle=False)

    model = TinyIQClassifier().to(device)
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        load_backbone(model, checkpoint_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss = float("nan")
    val_loss = float("nan")
    val_acc = float("nan")
    val_f1 = float("nan")
    test_loss = float("nan")
    test_acc = float("nan")
    test_f1 = float("nan")
    best_val_f1 = -1.0
    best_epoch = 0
    best_test_loss = float("nan")
    best_test_acc = float("nan")
    best_test_f1 = float("nan")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_classifier_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, num_classes=4)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device, num_classes=4)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_test_loss = test_loss
            best_test_acc = test_acc
            best_test_f1 = test_f1

        if not args.quiet:
            print(
                f"  {method} epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | "
                f"val_f1={val_f1:.4f} | test_acc={test_acc:.4f} | "
                f"test_f1={test_f1:.4f} | best_epoch={best_epoch}",
                flush=True,
            )

    return {
        "seed": seed,
        "method": method,
        "label_ratio": args.label_ratio,
        "labeled_train_samples": len(labeled_train_idx),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "best_test_loss": best_test_loss,
        "best_test_acc": best_test_acc,
        "best_test_macro_f1": best_test_f1,
    }


def print_summary(rows: list[dict]) -> None:
    print("\n=== Unified objective summary ===")
    methods = ["scratch", "noisy_clean_pretrained", "noisy_noisy_pretrained"]
    for metric_prefix in ["final", "best"]:
        print(f"\n[{metric_prefix}]")
        baseline_acc = None
        baseline_f1 = None
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            if metric_prefix == "final":
                acc = np.array([row["test_acc"] for row in method_rows], dtype=np.float64)
                f1 = np.array([row["test_macro_f1"] for row in method_rows], dtype=np.float64)
                loss = np.array([row["test_loss"] for row in method_rows], dtype=np.float64)
            else:
                acc = np.array([row["best_test_acc"] for row in method_rows], dtype=np.float64)
                f1 = np.array([row["best_test_macro_f1"] for row in method_rows], dtype=np.float64)
                loss = np.array([row["best_test_loss"] for row in method_rows], dtype=np.float64)
            if method == "scratch":
                baseline_acc = acc
                baseline_f1 = f1
            print(
                f"  {method:<24} | "
                f"test_acc={acc.mean():.4f} +/- {acc.std(ddof=0):.4f} | "
                f"test_macro_f1={f1.mean():.4f} +/- {f1.std(ddof=0):.4f} | "
                f"test_loss={loss.mean():.4f} +/- {loss.std(ddof=0):.4f}",
                flush=True,
            )
        for method in ["noisy_clean_pretrained", "noisy_noisy_pretrained"]:
            method_rows = [row for row in rows if row["method"] == method]
            if metric_prefix == "final":
                acc = np.array([row["test_acc"] for row in method_rows], dtype=np.float64)
                f1 = np.array([row["test_macro_f1"] for row in method_rows], dtype=np.float64)
            else:
                acc = np.array([row["best_test_acc"] for row in method_rows], dtype=np.float64)
                f1 = np.array([row["best_test_macro_f1"] for row in method_rows], dtype=np.float64)
            print(
                f"  delta {method}-scratch | "
                f"acc={(acc - baseline_acc).mean():+.4f} | "
                f"macro_f1={(f1 - baseline_f1).mean():+.4f}",
                flush=True,
            )


def save_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "method",
        "label_ratio",
        "labeled_train_samples",
        "train_loss",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "test_loss",
        "test_acc",
        "test_macro_f1",
        "best_epoch",
        "best_val_macro_f1",
        "best_test_loss",
        "best_test_acc",
        "best_test_macro_f1",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/iq_4mods_awgn_views_n20000.npz"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--label-ratio", type=float, default=0.1)
    parser.add_argument("--ssl-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ssl-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ssl-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--clean-checkpoint-dir", type=Path, default=Path("checkpoints/unified_clean"))
    parser.add_argument("--noisy-checkpoint-dir", type=Path, default=Path("checkpoints/unified_noisy"))
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/unified_objectives_label010.csv"))
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
    assert args.ssl_epochs > 0
    assert args.epochs > 0

    device = torch.device("cpu")
    cls_dataset = IQDataset(args.data)
    clean_dataset = IQDenoiseDataset(args.data)
    noisy_dataset = IQNoisyPairDataset(args.data)
    rows = []

    print("=== Unified objective comparison ===")
    print(f"Device: {device}")
    print(f"Data file: {args.data}")
    print(f"Seeds: {args.seeds}")
    print(f"Label ratio: {args.label_ratio}")
    print(f"SSL epochs: {args.ssl_epochs}")
    print(f"Finetune epochs: {args.epochs}")

    for seed in args.seeds:
        train_idx, val_idx, test_idx = stratified_split_indices(
            labels=cls_dataset.y,
            train_ratio=0.7,
            val_ratio=0.15,
            seed=seed,
        )
        labeled_train_idx = choose_labeled_subset(
            labels=cls_dataset.y,
            candidate_indices=train_idx,
            label_ratio=args.label_ratio,
            seed=seed,
        )

        print(f"\n--- Seed {seed} ---")
        print(f"Labeled train samples: {len(labeled_train_idx)}")
        clean_ckpt = pretrain_or_load_clean(clean_dataset, train_idx, val_idx, seed, args, device)
        noisy_ckpt = pretrain_or_load_noisy(noisy_dataset, train_idx, val_idx, seed, args, device)

        for method, ckpt in [
            ("scratch", None),
            ("noisy_clean_pretrained", clean_ckpt),
            ("noisy_noisy_pretrained", noisy_ckpt),
        ]:
            row = finetune_for_method(
                dataset=cls_dataset,
                labeled_train_idx=labeled_train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                seed=seed,
                method=method,
                checkpoint_path=ckpt,
                args=args,
                device=device,
            )
            rows.append(row)
            print(
                f"{method} test | acc={row['test_acc']:.4f} | "
                f"macro_f1={row['test_macro_f1']:.4f} | "
                f"best_epoch={row['best_epoch']} | "
                f"best_acc={row['best_test_acc']:.4f} | "
                f"best_f1={row['best_test_macro_f1']:.4f}",
                flush=True,
            )

    print_summary(rows)
    save_csv(rows, args.out)
    print(f"\nSaved results CSV: {args.out}")
    if args.log_file is not None:
        print(f"Saved live log: {args.log_file}")
    print("\nPASS: unified objective comparison completed.")

    if log_handle is not None:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main()
