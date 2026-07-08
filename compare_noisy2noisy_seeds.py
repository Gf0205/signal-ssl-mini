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
from device_utils import print_device_report, resolve_device, seed_cuda
from finetune import choose_labeled_subset, evaluate, train_one_epoch as train_classifier_one_epoch
from model import TinyIQClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_cuda(seed)


def make_loader(dataset, indices, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def load_backbone(model: TinyIQClassifier, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing noisy-to-noisy checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert "backbone" in checkpoint
    model.backbone.load_state_dict(checkpoint["backbone"], strict=True)


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
    method = "noisy2noisy_pretrained" if pretrained_path is not None else "scratch"
    if pretrained_path is not None:
        load_backbone(model, pretrained_path)

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
                f"train_loss={train_loss:.4f} | "
                f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | "
                f"test_acc={test_acc:.4f} | test_f1={test_f1:.4f} | "
                f"best_epoch={best_epoch}",
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
    print("\n=== Noisy-to-noisy multi-seed summary ===")
    for metric_prefix in ["final", "best"]:
        print(f"\n[{metric_prefix}]")
        for method in ["scratch", "noisy2noisy_pretrained"]:
            method_rows = [row for row in rows if row["method"] == method]
            if metric_prefix == "final":
                acc = np.array([row["test_acc"] for row in method_rows], dtype=np.float64)
                f1 = np.array([row["test_macro_f1"] for row in method_rows], dtype=np.float64)
                loss = np.array([row["test_loss"] for row in method_rows], dtype=np.float64)
            else:
                acc = np.array([row["best_test_acc"] for row in method_rows], dtype=np.float64)
                f1 = np.array([row["best_test_macro_f1"] for row in method_rows], dtype=np.float64)
                loss = np.array([row["best_test_loss"] for row in method_rows], dtype=np.float64)

            print(
                f"  {method:<24} | "
                f"test_acc={acc.mean():.4f} +/- {acc.std(ddof=0):.4f} | "
                f"test_macro_f1={f1.mean():.4f} +/- {f1.std(ddof=0):.4f} | "
                f"test_loss={loss.mean():.4f} +/- {loss.std(ddof=0):.4f}"
            )

        scratch_by_seed = {row["seed"]: row for row in rows if row["method"] == "scratch"}
        noisy_by_seed = {
            row["seed"]: row for row in rows if row["method"] == "noisy2noisy_pretrained"
        }
        delta_acc = []
        delta_f1 = []
        for seed in sorted(scratch_by_seed):
            if metric_prefix == "final":
                delta_acc.append(noisy_by_seed[seed]["test_acc"] - scratch_by_seed[seed]["test_acc"])
                delta_f1.append(
                    noisy_by_seed[seed]["test_macro_f1"] - scratch_by_seed[seed]["test_macro_f1"]
                )
            else:
                delta_acc.append(
                    noisy_by_seed[seed]["best_test_acc"] - scratch_by_seed[seed]["best_test_acc"]
                )
                delta_f1.append(
                    noisy_by_seed[seed]["best_test_macro_f1"]
                    - scratch_by_seed[seed]["best_test_macro_f1"]
                )
        print(
            f"  delta noisy2noisy-scratch | "
            f"acc={np.mean(delta_acc):+.4f} | macro_f1={np.mean(delta_f1):+.4f}"
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-prefix", type=str, default="noisy2noisy_backbone_seed")
    parser.add_argument("--out", type=Path, default=Path("results/noisy2noisy_n20000_label010.csv"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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

    device = resolve_device(args.device)
    dataset = IQDataset(args.data)
    rows = []

    print("=== Noisy-to-noisy downstream comparison ===")
    print_device_report(device)
    print(f"Data file: {args.data}")
    print(f"Seeds: {args.seeds}")
    print(f"Label ratio: {args.label_ratio}")
    print(f"Finetune epochs: {args.epochs}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

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
        ckpt_path = args.checkpoint_dir / f"{args.checkpoint_prefix}{seed}.pt"

        print(f"\n--- Seed {seed} ---")
        print(f"Labeled train samples: {len(labeled_train_idx)}")
        print(f"Noisy-to-noisy checkpoint: {ckpt_path}")

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
            f"macro_f1={scratch_row['test_macro_f1']:.4f} | "
            f"best_epoch={scratch_row['best_epoch']} | "
            f"best_acc={scratch_row['best_test_acc']:.4f} | "
            f"best_f1={scratch_row['best_test_macro_f1']:.4f}",
            flush=True,
        )

        noisy_row = finetune_for_seed(
            dataset=dataset,
            labeled_train_idx=labeled_train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=seed,
            args=args,
            device=device,
            pretrained_path=ckpt_path,
        )
        rows.append(noisy_row)
        print(
            f"Noisy2Noisy test | acc={noisy_row['test_acc']:.4f} | "
            f"macro_f1={noisy_row['test_macro_f1']:.4f} | "
            f"best_epoch={noisy_row['best_epoch']} | "
            f"best_acc={noisy_row['best_test_acc']:.4f} | "
            f"best_f1={noisy_row['best_test_macro_f1']:.4f}",
            flush=True,
        )

    print_summary(rows)
    save_csv(rows, args.out)
    print(f"\nSaved results CSV: {args.out}")
    if args.log_file is not None:
        print(f"Saved live log: {args.log_file}")
    print("\nPASS: noisy-to-noisy downstream comparison completed.")

    if log_handle is not None:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main()
