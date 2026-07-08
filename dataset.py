import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


MOD_NAMES = ["BPSK", "QPSK", "8PSK", "16QAM"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class IQDataset(Dataset):
    """Load IQ data saved by data.py.

    Each item is:
      x: float32 tensor with shape [2, L]
      y: int64 scalar label
    """

    def __init__(self, npz_path: Path) -> None:
        if not npz_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=False)
        self.x = data["x"].astype(np.float32)
        self.y = data["y"].astype(np.int64)
        self.snr = data["snr"].astype(np.float32)

        assert self.x.ndim == 3, f"Expected X shape [N, 2, L], got {self.x.shape}"
        assert self.x.shape[1] == 2, f"Expected IQ channel dim=2, got {self.x.shape}"
        assert self.y.shape == (self.x.shape[0],), f"Bad y shape: {self.y.shape}"
        assert self.snr.shape == (self.x.shape[0],), f"Bad snr shape: {self.snr.shape}"
        assert len(np.unique(self.y)) == len(MOD_NAMES), "Expected 4 modulation classes"
        assert not np.isnan(self.x).any(), "X contains NaN"
        assert not np.isinf(self.x).any(), "X contains Inf"

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[index])
        y = torch.tensor(self.y[index], dtype=torch.long)

        assert x.ndim == 2 and x.shape[0] == 2, f"Bad sample shape: {x.shape}"
        assert y.ndim == 0, f"Label should be scalar, got {y.shape}"
        return x, y


def stratified_split_indices(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    assert labels.ndim == 1
    assert 0.0 < train_ratio < 1.0
    assert 0.0 < val_ratio < 1.0
    assert train_ratio + val_ratio < 1.0

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for label in sorted(np.unique(labels).tolist()):
        class_indices = np.where(labels == label)[0]
        rng.shuffle(class_indices)

        n_total = len(class_indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_indices.extend(class_indices[:n_train].tolist())
        val_indices.extend(class_indices[n_train : n_train + n_val].tolist())
        test_indices.extend(class_indices[n_train + n_val :].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    assert len(set(train_indices) & set(val_indices)) == 0
    assert len(set(train_indices) & set(test_indices)) == 0
    assert len(set(val_indices) & set(test_indices)) == 0
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(labels)

    return train_indices, val_indices, test_indices


def print_label_counts(name: str, labels: np.ndarray, indices: list[int]) -> None:
    print(f"\n{name} split: n={len(indices)}")
    split_labels = labels[indices]
    for label, mod_name in enumerate(MOD_NAMES):
        count = int(np.sum(split_labels == label))
        print(f"  {label}: {mod_name:<5} count={count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/iq_4mods_awgn.npz"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert args.batch_size > 0
    set_seed(args.seed)

    dataset = IQDataset(args.data)
    print("=== Dataset loaded ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Raw X shape: {dataset.x.shape}  dtype: {dataset.x.dtype}")
    print(f"Raw y shape: {dataset.y.shape}  dtype: {dataset.y.dtype}")
    print(f"Raw snr shape: {dataset.snr.shape}  dtype: {dataset.snr.dtype}")

    train_idx, val_idx, test_idx = stratified_split_indices(
        labels=dataset.y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print_label_counts("Train", dataset.y, train_idx)
    print_label_counts("Val", dataset.y, val_idx)
    print_label_counts("Test", dataset.y, test_idx)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    x_batch, y_batch = next(iter(train_loader))
    print("\n=== One train batch ===")
    print(f"x_batch shape: {tuple(x_batch.shape)}  dtype: {x_batch.dtype}")
    print(f"y_batch shape: {tuple(y_batch.shape)}  dtype: {y_batch.dtype}")
    print(f"first 10 labels: {y_batch[:10].tolist()}")

    assert x_batch.ndim == 3, f"Expected batch X [B, 2, L], got {x_batch.shape}"
    assert x_batch.shape[1] == 2, f"Expected IQ channel dim=2, got {x_batch.shape}"
    assert y_batch.ndim == 1, f"Expected batch y [B], got {y_batch.shape}"
    assert x_batch.dtype == torch.float32
    assert y_batch.dtype == torch.long

    print("\nPASS: Dataset/DataLoader sanity checks completed.")


if __name__ == "__main__":
    main()
