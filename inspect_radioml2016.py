import argparse
import pickle
from pathlib import Path

import numpy as np


TARGET_MODS = ["BPSK", "QPSK", "8PSK", "QAM16"]


def normalize_key(key: tuple[object, object]) -> tuple[str, int]:
    mod, snr = key
    if isinstance(mod, bytes):
        mod = mod.decode("utf-8")
    return str(mod), int(snr)


def load_pickle(path: Path) -> dict[tuple[str, int], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"RadioML pickle not found: {path}")

    # Pickle can execute code during loading. Only inspect files from trusted
    # dataset mirrors such as Kaggle/Zenodo/official DeepSig.
    with path.open("rb") as f:
        raw = pickle.load(f, encoding="latin1")

    if not isinstance(raw, dict):
        raise TypeError(f"Expected a dict, got {type(raw)}")

    out = {}
    for key, value in raw.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Unexpected key format: {key!r}")
        out[normalize_key(key)] = np.asarray(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--show-examples", type=int, default=8)
    args = parser.parse_args()

    assert args.show_examples >= 0

    data = load_pickle(args.input)
    keys = sorted(data.keys(), key=lambda item: (item[0], item[1]))
    mods = sorted({mod for mod, _ in keys})
    snrs = sorted({snr for _, snr in keys})

    print("=== RadioML2016 pickle inspection ===")
    print(f"Input file: {args.input}")
    print(f"Total keys: {len(keys)}")
    print(f"Modulations ({len(mods)}): {mods}")
    print(f"SNRs ({len(snrs)}): {snrs}")

    if args.show_examples > 0:
        print("\nExample keys:")
        for key in keys[: args.show_examples]:
            print(f"  {key}: shape={data[key].shape} dtype={data[key].dtype}")

    print("\nTarget 4-class mapping check:")
    missing = []
    for mod in TARGET_MODS:
        mod_keys = [(m, snr) for m, snr in keys if m == mod]
        if not mod_keys:
            missing.append(mod)
            print(f"  {mod:<5}: MISSING")
            continue

        shapes = sorted({tuple(data[key].shape) for key in mod_keys})
        dtypes = sorted({str(data[key].dtype) for key in mod_keys})
        count = int(sum(data[key].shape[0] for key in mod_keys))
        print(
            f"  {mod:<5}: keys={len(mod_keys):2d} total_samples={count:6d} "
            f"shapes={shapes} dtypes={dtypes}"
        )

    if missing:
        raise KeyError(f"Missing target modulations: {missing}")

    print("\nShape validation:")
    bad_shapes = []
    for key in keys:
        shape = data[key].shape
        if len(shape) != 3 or shape[1] != 2 or shape[2] != 128:
            bad_shapes.append((key, shape))
    if bad_shapes:
        for key, shape in bad_shapes[:10]:
            print(f"  BAD {key}: shape={shape}")
        raise ValueError(f"Found {len(bad_shapes)} keys with non-[N, 2, 128] shape")

    print("  All arrays have shape [N, 2, 128].")
    print("\nPASS: RadioML2016 pickle inspection completed.")


if __name__ == "__main__":
    main()
