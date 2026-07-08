import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import IQDataset, MOD_NAMES, stratified_split_indices
from device_utils import print_device_report, resolve_device, seed_cuda
from model import TinyIQClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_cuda(seed)


def choose_labeled_subset(
    labels: np.ndarray,
    candidate_indices: list[int],
    label_ratio: float,
    seed: int,
) -> list[int]:
    assert 0.0 < label_ratio <= 1.0

    rng = np.random.default_rng(seed)
    candidate_indices_array = np.array(candidate_indices, dtype=np.int64)
    labeled_indices = []

    for label in sorted(np.unique(labels).tolist()):
        class_indices = candidate_indices_array[labels[candidate_indices_array] == label]
        rng.shuffle(class_indices)

        n_labeled = max(1, int(round(len(class_indices) * label_ratio)))
        labeled_indices.extend(class_indices[:n_labeled].tolist())

    rng.shuffle(labeled_indices)
    return labeled_indices


def print_split_counts(name: str, labels: np.ndarray, indices: list[int]) -> None:
    print(f"\n{name}: n={len(indices)}")
    split_labels = labels[indices]
    for label, mod_name in enumerate(MOD_NAMES):
        count = int(np.sum(split_labels == label))
        print(f"  {label}: {mod_name:<5} count={count}")


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1_values = []

    for cls in range(num_classes):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_values.append(f1)

    return float(np.mean(f1_values))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            pred = torch.argmax(logits, dim=1)

            batch_size = x.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_true.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    avg_loss = total_loss / total_samples
    accuracy = float(np.mean(y_true == y_pred))
    macro_f1 = macro_f1_score(y_true, y_pred, num_classes=num_classes)
    return avg_loss, accuracy, macro_f1


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def load_pretrained_backbone(model: TinyIQClassifier, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert "backbone" in checkpoint, "Checkpoint must contain a 'backbone' state_dict"
    missing_keys, unexpected_keys = model.backbone.load_state_dict(checkpoint["backbone"], strict=True)
    assert len(missing_keys) == 0, f"Missing keys when loading backbone: {missing_keys}"
    assert len(unexpected_keys) == 0, f"Unexpected keys when loading backbone: {unexpected_keys}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/iq_4mods_awgn.npz"))
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--label-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    assert args.epochs > 0
    assert args.batch_size > 0
    assert args.lr > 0
    assert args.weight_decay >= 0
    assert 0.0 < args.label_ratio <= 1.0

    set_seed(args.seed)
    device = resolve_device(args.device)

    dataset = IQDataset(args.data)
    train_idx, val_idx, test_idx = stratified_split_indices(
        labels=dataset.y,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=args.seed,
    )
    labeled_train_idx = choose_labeled_subset(
        labels=dataset.y,
        candidate_indices=train_idx,
        label_ratio=args.label_ratio,
        seed=args.seed,
    )

    run_name = "SSL-pretrained finetune" if args.pretrained is not None else "Scratch finetune"

    print(f"=== {run_name} setup ===")
    print_device_report(device)
    print(f"Data file: {args.data}")
    print(f"Pretrained checkpoint: {args.pretrained}")
    print(f"Label ratio within train split: {args.label_ratio}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    print_split_counts("Labeled train split", dataset.y, labeled_train_idx)
    print_split_counts("Val split", dataset.y, val_idx)
    print_split_counts("Test split", dataset.y, test_idx)

    train_loader = DataLoader(
        Subset(dataset, labeled_train_idx),
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
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    x_batch, y_batch = next(iter(train_loader))
    print("\n=== First labeled train batch ===")
    print(f"x_batch shape: {tuple(x_batch.shape)}  dtype: {x_batch.dtype}")
    print(f"y_batch shape: {tuple(y_batch.shape)}  dtype: {y_batch.dtype}")
    x_device_check = x_batch.to(device)
    print(f"x_batch device after .to(device): {x_device_check.device}")
    assert x_batch.ndim == 3 and x_batch.shape[1] == 2
    assert y_batch.ndim == 1

    model = TinyIQClassifier().to(device)
    if args.pretrained is not None:
        load_pretrained_backbone(model, args.pretrained)
        print(f"\nLoaded pretrained backbone from: {args.pretrained}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"\n=== Training: {run_name} ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, num_classes=4)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
        )

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device, num_classes=4)
    print(f"\n=== Final {run_name} on test split ===")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc:.4f}")
    print(f"test_macro_f1={test_f1:.4f}")
    print(f"\nPASS: {run_name} completed.")


if __name__ == "__main__":
    main()
