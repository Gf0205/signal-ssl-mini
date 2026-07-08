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


def load_backbone(model: TinyIQClassifier, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing denoising checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert "backbone" in checkpoint
    model.backbone.load_state_dict(checkpoint["backbone"], strict=True)


def finetune_with_history(
    dataset: IQDataset,
    labeled_train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    seed: int,
    method: str,
    max_epochs: int,
    args,
    device: torch.device,
    pretrained_path: Path | None,
) -> dict:
    set_seed(seed)
    train_loader = make_loader(dataset, labeled_train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(dataset, val_idx, args.batch_size, shuffle=False)
    test_loader = make_loader(dataset, test_idx, args.batch_size, shuffle=False)

    model = TinyIQClassifier().to(device)
    if pretrained_path is not None:
        load_backbone(model, pretrained_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history_rows = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_test_loss = float("nan")
    best_test_acc = float("nan")
    best_test_f1 = float("nan")

    for epoch in range(1, max_epochs + 1):
        train_loss = train_classifier_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, num_classes=4)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device, num_classes=4)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_test_loss = test_loss
            best_test_acc = test_acc
            best_test_f1 = test_f1

        history_rows.append(
            {
                "seed": seed,
                "method": method,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_macro_f1": test_f1,
                "best_epoch_so_far": best_epoch,
                "best_val_macro_f1_so_far": best_val_f1,
                "best_test_acc_so_far": best_test_acc,
                "best_test_macro_f1_so_far": best_test_f1,
            }
        )

        if not args.quiet:
            print(
                f"  {method} epoch {epoch:02d}/{max_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | "
                f"test_acc={test_acc:.4f} | test_f1={test_f1:.4f} | "
                f"best_epoch={best_epoch}",
                flush=True,
            )

    final_row = history_rows[-1]
    summary_row = {
        "seed": seed,
        "method": method,
        "max_epochs": max_epochs,
        "final_test_loss": final_row["test_loss"],
        "final_test_acc": final_row["test_acc"],
        "final_test_macro_f1": final_row["test_macro_f1"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "best_test_loss": best_test_loss,
        "best_test_acc": best_test_acc,
        "best_test_macro_f1": best_test_f1,
    }
    return {"summary": summary_row, "history": history_rows}


def save_csv(rows: list[dict], out_path: Path, fieldnames: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(summary_rows: list[dict], epoch_budgets: list[int]) -> None:
    print("\n=== Epoch-budget summary ===")
    for max_epochs in epoch_budgets:
        print(f"\nMax epochs = {max_epochs}")
        budget_rows = [row for row in summary_rows if row["max_epochs"] == max_epochs]
        for metric_prefix in ["final", "best"]:
            print(f"  [{metric_prefix}]")
            for method in ["scratch", "denoise_pretrained"]:
                method_rows = [row for row in budget_rows if row["method"] == method]
                acc = np.array([row[f"{metric_prefix}_test_acc"] for row in method_rows])
                f1 = np.array([row[f"{metric_prefix}_test_macro_f1"] for row in method_rows])
                print(
                    f"    {method:<20} | "
                    f"acc={acc.mean():.4f} +/- {acc.std(ddof=0):.4f} | "
                    f"macro_f1={f1.mean():.4f} +/- {f1.std(ddof=0):.4f}"
                )

            scratch_by_seed = {
                row["seed"]: row for row in budget_rows if row["method"] == "scratch"
            }
            denoise_by_seed = {
                row["seed"]: row for row in budget_rows if row["method"] == "denoise_pretrained"
            }
            delta_acc = []
            delta_f1 = []
            for seed in sorted(scratch_by_seed):
                delta_acc.append(
                    denoise_by_seed[seed][f"{metric_prefix}_test_acc"]
                    - scratch_by_seed[seed][f"{metric_prefix}_test_acc"]
                )
                delta_f1.append(
                    denoise_by_seed[seed][f"{metric_prefix}_test_macro_f1"]
                    - scratch_by_seed[seed][f"{metric_prefix}_test_macro_f1"]
                )
            print(
                f"    delta denoise-scratch | "
                f"acc={np.mean(delta_acc):+.4f} | macro_f1={np.mean(delta_f1):+.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/iq_4mods_awgn_clean_n20000.npz"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--label-ratio", type=float, default=0.2)
    parser.add_argument("--epoch-budgets", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/label_ratio_denoise_n20000"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/epoch_budget_label020_summary.csv"),
    )
    parser.add_argument(
        "--history-out",
        type=Path,
        default=Path("results/epoch_budget_label020_history.csv"),
    )
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
    assert len(args.epoch_budgets) > 0
    assert all(epoch > 0 for epoch in args.epoch_budgets)

    device = torch.device("cpu")
    dataset = IQDataset(args.data)
    summary_rows = []
    history_rows = []

    print("=== Epoch-budget fairness check ===")
    print(f"Device: {device}")
    print(f"Data file: {args.data}")
    print(f"Seeds: {args.seeds}")
    print(f"Label ratio: {args.label_ratio}")
    print(f"Epoch budgets: {args.epoch_budgets}")
    print(f"Denoising checkpoint dir: {args.checkpoint_dir}")

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
        denoise_ckpt = args.checkpoint_dir / f"denoise_backbone_seed{seed}.pt"

        print(f"\n--- Seed {seed} ---")
        print(f"Labeled train samples: {len(labeled_train_idx)}")
        print(f"Denoise checkpoint: {denoise_ckpt}")

        for max_epochs in args.epoch_budgets:
            print(f"\n  Epoch budget = {max_epochs}", flush=True)
            for method, pretrained_path in [
                ("scratch", None),
                ("denoise_pretrained", denoise_ckpt),
            ]:
                result = finetune_with_history(
                    dataset=dataset,
                    labeled_train_idx=labeled_train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    seed=seed,
                    method=method,
                    max_epochs=max_epochs,
                    args=args,
                    device=device,
                    pretrained_path=pretrained_path,
                )
                summary_rows.append(result["summary"])
                history_rows.extend(
                    {
                        **row,
                        "max_epochs": max_epochs,
                        "label_ratio": args.label_ratio,
                    }
                    for row in result["history"]
                )
                row = result["summary"]
                print(
                    f"  {method} done | "
                    f"final_acc={row['final_test_acc']:.4f} | "
                    f"final_f1={row['final_test_macro_f1']:.4f} | "
                    f"best_epoch={row['best_epoch']} | "
                    f"best_test_acc={row['best_test_acc']:.4f} | "
                    f"best_test_f1={row['best_test_macro_f1']:.4f}",
                    flush=True,
                )

    summary_fields = [
        "seed",
        "method",
        "max_epochs",
        "final_test_loss",
        "final_test_acc",
        "final_test_macro_f1",
        "best_epoch",
        "best_val_macro_f1",
        "best_test_loss",
        "best_test_acc",
        "best_test_macro_f1",
    ]
    history_fields = [
        "seed",
        "label_ratio",
        "method",
        "max_epochs",
        "epoch",
        "train_loss",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "test_loss",
        "test_acc",
        "test_macro_f1",
        "best_epoch_so_far",
        "best_val_macro_f1_so_far",
        "best_test_acc_so_far",
        "best_test_macro_f1_so_far",
    ]

    print_summary(summary_rows, args.epoch_budgets)
    save_csv(summary_rows, args.out, summary_fields)
    save_csv(history_rows, args.history_out, history_fields)

    print(f"\nSaved summary CSV: {args.out}")
    print(f"Saved history CSV: {args.history_out}")
    if args.log_file is not None:
        print(f"Saved live log: {args.log_file}")
    print("\nPASS: epoch-budget fairness check script completed.")

    if log_handle is not None:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main()
