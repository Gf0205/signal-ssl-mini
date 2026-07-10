import argparse
import pickle
import random
from pathlib import Path

import numpy as np


TARGET_MODS = ["BPSK", "QPSK", "8PSK", "QAM16"]
OUTPUT_MOD_NAMES = ["BPSK", "QPSK", "8PSK", "16QAM"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_key(key: tuple[object, object]) -> tuple[str, int]:
    mod, snr = key
    if isinstance(mod, bytes):
        mod = mod.decode("utf-8")
    return str(mod), int(snr)


def ensure_iq_shape(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected RadioML block shape [N, 2, L], got {x.shape}")
    if x.shape[1] == 2:
        return x
    if x.shape[2] == 2:
        return np.transpose(x, (0, 2, 1)).copy()
    raise ValueError(f"Cannot find IQ channel dimension in shape {x.shape}")


def rms_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    power = np.mean(x**2, axis=(1, 2), keepdims=True)
    return x / np.sqrt(power + eps)


def add_awgn_views(
    x: np.ndarray,
    snr_min: float,
    snr_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if snr_min > snr_max:
        raise ValueError("view_snr_min must be <= view_snr_max")

    n = x.shape[0]
    snr_a = rng.uniform(snr_min, snr_max, size=n).astype(np.float32)
    snr_b = rng.uniform(snr_min, snr_max, size=n).astype(np.float32)

    signal_power = np.mean(x**2, axis=(1, 2), keepdims=True)
    noise_power_a = signal_power / (10.0 ** (snr_a[:, None, None] / 10.0))
    noise_power_b = signal_power / (10.0 ** (snr_b[:, None, None] / 10.0))

    noise_a = rng.normal(0.0, np.sqrt(noise_power_a), size=x.shape).astype(np.float32)
    noise_b = rng.normal(0.0, np.sqrt(noise_power_b), size=x.shape).astype(np.float32)
    return (x + noise_a).astype(np.float32), (x + noise_b).astype(np.float32), snr_a, snr_b


def load_radioml_pickle(path: Path) -> dict[tuple[str, int], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"RadioML pickle not found: {path}")

    # RadioML 2016.10A is commonly distributed as a Python pickle created with
    # Python 2. Only load files from a trusted source.
    with path.open("rb") as f:
        raw = pickle.load(f, encoding="latin1")

    if not isinstance(raw, dict):
        raise TypeError(f"Expected pickle to contain a dict, got {type(raw)}")

    normalized = {}
    for key, value in raw.items():
        normalized[normalize_key(key)] = ensure_iq_shape(value)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/radioml2016_4mods.npz"))
    parser.add_argument("--snr-min", type=int, default=None)
    parser.add_argument("--snr-max", type=int, default=None)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--make-views", action="store_true")
    parser.add_argument(
        "--view-mode",
        choices=["awgn", "identity"],
        default="awgn",
        help="AWGN creates two added-noise views; identity pairs each received sample with itself.",
    )
    parser.add_argument("--view-snr-min", type=float, default=10.0)
    parser.add_argument("--view-snr-max", type=float, default=20.0)
    args = parser.parse_args()

    assert args.max_per_class >= 0
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = load_radioml_pickle(args.input)
    available_mods = sorted({mod for mod, _ in data.keys()})
    available_snrs = sorted({snr for _, snr in data.keys()})

    missing_mods = [mod for mod in TARGET_MODS if mod not in available_mods]
    if missing_mods:
        raise KeyError(f"Missing required mods in RadioML file: {missing_mods}")

    x_blocks = []
    y_blocks = []
    snr_blocks = []

    print("=== RadioML2016.10A conversion ===")
    print(f"Input file: {args.input}")
    print(f"Available mods: {available_mods}")
    print(f"Available SNRs: {available_snrs}")
    print(f"Selected source mods: {TARGET_MODS}")
    print(f"Output mod names: {OUTPUT_MOD_NAMES}")

    for label, mod in enumerate(TARGET_MODS):
        class_blocks = []
        class_snrs = []
        for snr in available_snrs:
            if args.snr_min is not None and snr < args.snr_min:
                continue
            if args.snr_max is not None and snr > args.snr_max:
                continue
            key = (mod, snr)
            if key not in data:
                continue
            block = data[key]
            class_blocks.append(block)
            class_snrs.append(np.full(block.shape[0], snr, dtype=np.float32))

        if not class_blocks:
            raise ValueError(f"No samples selected for modulation {mod}")

        x_class = np.concatenate(class_blocks, axis=0).astype(np.float32)
        snr_class = np.concatenate(class_snrs, axis=0).astype(np.float32)

        if args.max_per_class > 0 and x_class.shape[0] > args.max_per_class:
            indices = rng.permutation(x_class.shape[0])[: args.max_per_class]
            x_class = x_class[indices]
            snr_class = snr_class[indices]

        y_class = np.full(x_class.shape[0], label, dtype=np.int64)
        x_blocks.append(x_class)
        y_blocks.append(y_class)
        snr_blocks.append(snr_class)
        print(f"{label}: {OUTPUT_MOD_NAMES[label]:<5} from {mod:<5} count={x_class.shape[0]}")

    x = np.concatenate(x_blocks, axis=0).astype(np.float32)
    y = np.concatenate(y_blocks, axis=0).astype(np.int64)
    snr = np.concatenate(snr_blocks, axis=0).astype(np.float32)

    indices = rng.permutation(x.shape[0])
    x = x[indices]
    y = y[indices]
    snr = snr[indices]

    if args.normalize:
        x = rms_normalize(x).astype(np.float32)

    save_data = {
        "x": x,
        "y": y,
        "snr": snr,
        "mod_names": np.asarray(OUTPUT_MOD_NAMES),
        "source_mod_names": np.asarray(TARGET_MODS),
    }

    if args.make_views:
        if args.view_mode == "awgn":
            x_noisy_a, x_noisy_b, snr_a, snr_b = add_awgn_views(
                x=x,
                snr_min=args.view_snr_min,
                snr_max=args.view_snr_max,
                rng=rng,
            )
        else:
            x_noisy_a = x.copy()
            x_noisy_b = x.copy()
            snr_a = snr.copy()
            snr_b = snr.copy()
        save_data.update(
            {
                "x_noisy_a": x_noisy_a,
                "x_noisy_b": x_noisy_b,
                "snr_a": snr_a,
                "snr_b": snr_b,
                "view_mode": np.asarray(args.view_mode),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **save_data)

    print("\n=== Saved converted dataset ===")
    print(f"Output file: {args.out}")
    print(f"x shape: {x.shape} dtype={x.dtype}")
    print(f"y shape: {y.shape} dtype={y.dtype}")
    print(f"snr shape: {snr.shape} dtype={snr.dtype}")
    if args.make_views:
        print(f"view mode: {args.view_mode}")
        print(f"x_noisy_a shape: {save_data['x_noisy_a'].shape}")
        print(f"x_noisy_b shape: {save_data['x_noisy_b'].shape}")
        if args.view_mode == "awgn":
            print(f"added-noise view SNR range: [{args.view_snr_min}, {args.view_snr_max}] dB")
        else:
            print(f"source SNR range: [{float(snr.min())}, {float(snr.max())}] dB")
    for label, name in enumerate(OUTPUT_MOD_NAMES):
        print(f"label {label}: {name:<5} count={int(np.sum(y == label))}")

    assert x.ndim == 3 and x.shape[1] == 2, f"Bad x shape: {x.shape}"
    assert y.shape == (x.shape[0],)
    assert snr.shape == (x.shape[0],)
    assert len(np.unique(y)) == len(OUTPUT_MOD_NAMES)
    assert np.isfinite(x).all()
    if args.make_views:
        assert save_data["x_noisy_a"].shape == x.shape
        assert save_data["x_noisy_b"].shape == x.shape
        assert np.isfinite(save_data["x_noisy_a"]).all()
        assert np.isfinite(save_data["x_noisy_b"]).all()
        if args.view_mode == "identity":
            assert np.array_equal(save_data["x_noisy_a"], x)
            assert np.array_equal(save_data["x_noisy_b"], x)
    print("\nPASS: RadioML conversion completed.")


if __name__ == "__main__":
    main()
