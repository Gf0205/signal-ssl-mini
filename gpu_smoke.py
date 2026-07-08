import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import IQDataset
from device_utils import print_device_report, resolve_device
from model import TinyIQClassifier, count_parameters
from pretrain import patchify_iq
from pretrain_noisy2noisy import (
    IQNoisyPairDataset,
    NoisyToNoisyIQPretrainer,
    reconstruction_loss,
)


def cuda_memory_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return allocated, reserved


def write_log(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/smoke_gpu_views.npz"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-out", type=Path, default=Path("checkpoints/gpu_smoke.pt"))
    parser.add_argument("--csv-out", type=Path, default=Path("results/gpu_smoke.csv"))
    parser.add_argument("--log-out", type=Path, default=Path("logs/gpu_smoke.txt"))
    args = parser.parse_args()

    assert args.batch_size > 0
    if not args.data.exists():
        raise FileNotFoundError(
            f"Missing smoke data: {args.data}. Generate it with data.py before running this script."
        )

    device = resolve_device(args.device)
    log_lines = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    log("=== GPU smoke test ===")
    print_device_report(device)
    log(f"Data file: {args.data}")

    cls_dataset = IQDataset(args.data)
    pair_dataset = IQNoisyPairDataset(args.data)
    n = min(len(cls_dataset), max(args.batch_size, 8))
    subset_idx = list(range(n))

    cls_loader = DataLoader(
        Subset(cls_dataset, subset_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    pair_loader = DataLoader(
        Subset(pair_dataset, subset_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    x, y = next(iter(cls_loader))
    log(f"Classifier batch CPU shape: x={tuple(x.shape)}, y={tuple(y.shape)}")
    x = x.to(device)
    y = y.to(device)
    log(f"Classifier batch device: x={x.device}, y={y.device}")
    assert x.device.type == device.type
    assert y.device.type == device.type

    classifier = TinyIQClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
    logits = classifier(x)
    cls_loss = criterion(logits, y)
    optimizer.zero_grad(set_to_none=True)
    cls_loss.backward()
    optimizer.step()
    log(f"Classifier logits shape: {tuple(logits.shape)}")
    log(f"Classifier loss after one backward: {float(cls_loss.item()):.6f}")

    x_noisy_a, x_noisy_b = next(iter(pair_loader))
    log(
        "Noisy pair batch CPU shape: "
        f"a={tuple(x_noisy_a.shape)}, b={tuple(x_noisy_b.shape)}"
    )
    x_noisy_a = x_noisy_a.to(device)
    x_noisy_b = x_noisy_b.to(device)
    log(f"Noisy pair batch device: a={x_noisy_a.device}, b={x_noisy_b.device}")
    assert x_noisy_a.device.type == device.type
    assert x_noisy_b.device.type == device.type

    pretrainer = NoisyToNoisyIQPretrainer().to(device)
    pre_optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=1e-3)
    target_patches = patchify_iq(x_noisy_b, patch_size=8)
    pred_patches = pretrainer(x_noisy_a)
    ssl_loss = reconstruction_loss(pred_patches, target_patches)
    pre_optimizer.zero_grad(set_to_none=True)
    ssl_loss.backward()
    pre_optimizer.step()
    log(f"Target patches shape: {tuple(target_patches.shape)}")
    log(f"Predicted patches shape: {tuple(pred_patches.shape)}")
    log(f"Noisy2Noisy loss after one backward: {float(ssl_loss.item()):.6f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    allocated_mb, reserved_mb = cuda_memory_mb()
    log(f"CUDA memory allocated MB: {allocated_mb:.2f}")
    log(f"CUDA memory reserved MB: {reserved_mb:.2f}")

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier": classifier.state_dict(),
            "noisy2noisy_backbone": pretrainer.backbone.state_dict(),
            "device": str(device),
        },
        args.checkpoint_out,
    )
    log(f"Saved checkpoint: {args.checkpoint_out}")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "device",
                "classifier_params",
                "pretrainer_params",
                "classifier_loss",
                "noisy2noisy_loss",
                "cuda_allocated_mb",
                "cuda_reserved_mb",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "device": str(device),
                "classifier_params": count_parameters(classifier),
                "pretrainer_params": count_parameters(pretrainer),
                "classifier_loss": float(cls_loss.item()),
                "noisy2noisy_loss": float(ssl_loss.item()),
                "cuda_allocated_mb": allocated_mb,
                "cuda_reserved_mb": reserved_mb,
            }
        )
    log(f"Saved CSV: {args.csv_out}")

    write_log(log_lines, args.log_out)
    print(f"Saved log: {args.log_out}", flush=True)
    print("\nPASS: GPU smoke test completed.", flush=True)


if __name__ == "__main__":
    main()
