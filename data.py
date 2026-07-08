import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np


MOD_NAMES = ["BPSK", "QPSK", "8PSK", "16QAM"]
LABEL_TO_MOD = {i: name for i, name in enumerate(MOD_NAMES)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def modulate_symbols(mod_name: str, length: int):
    """Return one complex baseband IQ sequence with shape [L]."""
    if mod_name == "BPSK":
        bits = np.random.randint(0, 2, size=length)
        symbols = 2 * bits - 1
        return symbols.astype(np.complex64)

    if mod_name == "QPSK":
        idx = np.random.randint(0, 4, size=length)
        phase = np.pi / 4 + idx * np.pi / 2
        return np.exp(1j * phase).astype(np.complex64)

    if mod_name == "8PSK":
        idx = np.random.randint(0, 8, size=length)
        phase = idx * 2 * np.pi / 8
        return np.exp(1j * phase).astype(np.complex64)

    if mod_name == "16QAM":
        levels = np.array([-3, -1, 1, 3], dtype=np.float32)
        i = np.random.choice(levels, size=length)
        q = np.random.choice(levels, size=length)
        symbols = (i + 1j * q) / np.sqrt(10.0)
        return symbols.astype(np.complex64)

    raise ValueError(f"Unknown modulation: {mod_name}")


def add_awgn(x, snr_db: float):
    """Add complex AWGN to a complex sequence x with shape [L]."""
    assert np.iscomplexobj(x), "x must be a complex IQ sequence"

    signal_power = np.mean(np.abs(x) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
    return (x + noise).astype(np.complex64)


def apply_phase_offset(x, phase_rad: float):
    """Apply a constant phase rotation to one complex IQ sequence."""
    assert np.iscomplexobj(x), "x must be a complex IQ sequence"
    return (x * np.exp(1j * phase_rad)).astype(np.complex64)


def apply_cfo(x, normalized_cfo: float):
    """Apply normalized CFO as a linear phase drift across IQ samples."""
    assert np.iscomplexobj(x), "x must be a complex IQ sequence"
    n = np.arange(x.shape[0], dtype=np.float32)
    drift = np.exp(1j * 2 * np.pi * normalized_cfo * n)
    return (x * drift).astype(np.complex64)


def complex_to_iq(x):
    """Convert complex sequence [L] to real-valued IQ array [2, L]."""
    assert x.ndim == 1, f"Expected [L], got {x.shape}"
    iq = np.stack([x.real, x.imag], axis=0).astype(np.float32)
    assert iq.shape == (2, x.shape[0])
    return iq


def generate_dataset(
    samples_per_class: int,
    length: int,
    snr_min: float,
    snr_max: float,
    phase_offset: bool,
    phase_max_deg: float,
    cfo_offset: bool,
    cfo_max: float,
) -> Tuple[object, object, object]:
    xs_a = []
    xs_b = []
    xs_clean = []
    ys = []
    snrs_a = []
    snrs_b = []
    phases_a = []
    phases_b = []
    cfo_a_values = []
    cfo_b_values = []
    phase_max_rad = np.deg2rad(phase_max_deg)

    for label, mod_name in LABEL_TO_MOD.items():
        for _ in range(samples_per_class):
            clean = modulate_symbols(mod_name, length)
            snr_a_db = np.random.uniform(snr_min, snr_max)
            snr_b_db = np.random.uniform(snr_min, snr_max)
            phase_a = np.random.uniform(-phase_max_rad, phase_max_rad) if phase_offset else 0.0
            phase_b = np.random.uniform(-phase_max_rad, phase_max_rad) if phase_offset else 0.0
            cfo_a = np.random.uniform(-cfo_max, cfo_max) if cfo_offset else 0.0
            cfo_b = np.random.uniform(-cfo_max, cfo_max) if cfo_offset else 0.0
            view_a = apply_phase_offset(clean, phase_a)
            view_b = apply_phase_offset(clean, phase_b)
            view_a = apply_cfo(view_a, cfo_a)
            view_b = apply_cfo(view_b, cfo_b)
            noisy_a = add_awgn(view_a, snr_a_db)
            noisy_b = add_awgn(view_b, snr_b_db)

            xs_a.append(complex_to_iq(noisy_a))
            xs_b.append(complex_to_iq(noisy_b))
            xs_clean.append(complex_to_iq(clean))
            ys.append(label)
            snrs_a.append(snr_a_db)
            snrs_b.append(snr_b_db)
            phases_a.append(phase_a)
            phases_b.append(phase_b)
            cfo_a_values.append(cfo_a)
            cfo_b_values.append(cfo_b)

    x = np.stack(xs_a, axis=0).astype(np.float32)
    x_noisy_b = np.stack(xs_b, axis=0).astype(np.float32)
    x_clean = np.stack(xs_clean, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    snr = np.array(snrs_a, dtype=np.float32)
    snr_b = np.array(snrs_b, dtype=np.float32)
    phase_a = np.array(phases_a, dtype=np.float32)
    phase_b = np.array(phases_b, dtype=np.float32)
    cfo_a = np.array(cfo_a_values, dtype=np.float32)
    cfo_b = np.array(cfo_b_values, dtype=np.float32)

    perm = np.random.permutation(len(y))
    x = x[perm]
    x_noisy_b = x_noisy_b[perm]
    x_clean = x_clean[perm]
    y = y[perm]
    snr = snr[perm]
    snr_b = snr_b[perm]
    phase_a = phase_a[perm]
    phase_b = phase_b[perm]
    cfo_a = cfo_a[perm]
    cfo_b = cfo_b[perm]

    assert x.shape == (samples_per_class * len(MOD_NAMES), 2, length)
    assert x_noisy_b.shape == x.shape
    assert x_clean.shape == x.shape
    assert y.shape == (x.shape[0],)
    assert snr.shape == (x.shape[0],)
    assert snr_b.shape == (x.shape[0],)
    assert phase_a.shape == (x.shape[0],)
    assert phase_b.shape == (x.shape[0],)
    assert cfo_a.shape == (x.shape[0],)
    assert cfo_b.shape == (x.shape[0],)
    assert x.dtype == np.float32
    assert x_noisy_b.dtype == np.float32
    assert x_clean.dtype == np.float32
    assert y.dtype == np.int64

    return x, x_noisy_b, x_clean, y, snr, snr_b, phase_a, phase_b, cfo_a, cfo_b


def print_dataset_report(
    x,
    y,
    snr,
    x_clean=None,
    x_noisy_b=None,
    snr_b=None,
    phase_a=None,
    phase_b=None,
    cfo_a=None,
    cfo_b=None,
) -> None:
    print("=== Dataset report ===")
    print(f"X shape: {x.shape}  dtype: {x.dtype}")
    if x_noisy_b is not None:
        print(f"X noisy B shape: {x_noisy_b.shape}  dtype: {x_noisy_b.dtype}")
    if x_clean is not None:
        print(f"X clean shape: {x_clean.shape}  dtype: {x_clean.dtype}")
    print(f"y shape: {y.shape}  dtype: {y.dtype}")
    print(f"snr shape: {snr.shape}  dtype: {snr.dtype}")
    print(f"One sample shape: {x[0].shape}  meaning: [2, L]")
    print(f"I channel shape: {x[0, 0].shape}")
    print(f"Q channel shape: {x[0, 1].shape}")
    print(f"SNR range: {snr.min():.2f} dB to {snr.max():.2f} dB")
    if snr_b is not None:
        print(f"SNR B range: {snr_b.min():.2f} dB to {snr_b.max():.2f} dB")
    if phase_a is not None and phase_b is not None:
        print(
            "Phase A range: "
            f"{np.rad2deg(phase_a.min()):.2f} deg to {np.rad2deg(phase_a.max()):.2f} deg"
        )
        print(
            "Phase B range: "
            f"{np.rad2deg(phase_b.min()):.2f} deg to {np.rad2deg(phase_b.max()):.2f} deg"
        )
    if cfo_a is not None and cfo_b is not None:
        print(f"CFO A range: {cfo_a.min():.5f} to {cfo_a.max():.5f}")
        print(f"CFO B range: {cfo_b.min():.5f} to {cfo_b.max():.5f}")

    print("\nLabel distribution:")
    for label, mod_name in LABEL_TO_MOD.items():
        count = int(np.sum(y == label))
        print(f"  {label}: {mod_name:<5} count={count}")

    assert x.ndim == 3, "X must be [N, 2, L]"
    assert x.shape[1] == 2, "Second dimension must be I/Q channels"
    assert len(np.unique(y)) == len(MOD_NAMES), "All modulation classes must appear"
    assert not np.isnan(x).any(), "X contains NaN"
    assert not np.isinf(x).any(), "X contains Inf"
    if x_noisy_b is not None:
        assert x_noisy_b.shape == x.shape, "x_noisy_b must match x shape"
        assert not np.isnan(x_noisy_b).any(), "x_noisy_b contains NaN"
        assert not np.isinf(x_noisy_b).any(), "x_noisy_b contains Inf"
    if x_clean is not None:
        assert x_clean.shape == x.shape, "x_clean must match x shape"
        assert not np.isnan(x_clean).any(), "x_clean contains NaN"
        assert not np.isinf(x_clean).any(), "x_clean contains Inf"


def save_debug_plot(x, y, out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        html_path = out_path.with_suffix(".html")
        save_debug_html(x, y, html_path)
        return

    fig, axes = plt.subplots(len(MOD_NAMES), 2, figsize=(10, 10))

    for label, mod_name in LABEL_TO_MOD.items():
        idx = int(np.where(y == label)[0][0])
        iq = x[idx]
        i = iq[0]
        q = iq[1]

        axes[label, 0].plot(i, label="I", linewidth=1.0)
        axes[label, 0].plot(q, label="Q", linewidth=1.0)
        axes[label, 0].set_title(f"{mod_name} waveform")
        axes[label, 0].set_xlim(0, min(64, iq.shape[1] - 1))
        axes[label, 0].grid(True, alpha=0.3)
        axes[label, 0].legend(loc="upper right")

        axes[label, 1].scatter(i, q, s=8, alpha=0.7)
        axes[label, 1].set_title(f"{mod_name} constellation")
        axes[label, 1].set_xlabel("I")
        axes[label, 1].set_ylabel("Q")
        axes[label, 1].axis("equal")
        axes[label, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved debug plot: {out_path}")


def points_to_polyline(values, width: int, height: int, limit: int = 64) -> str:
    values = np.asarray(values[:limit], dtype=np.float32)
    v_min = float(values.min())
    v_max = float(values.max())
    if abs(v_max - v_min) < 1e-6:
        v_max = v_min + 1.0

    points = []
    for idx, value in enumerate(values):
        px = idx / max(1, limit - 1) * width
        py = height - (float(value) - v_min) / (v_max - v_min) * height
        points.append(f"{px:.1f},{py:.1f}")
    return " ".join(points)


def constellation_points(i, q, width: int, height: int) -> str:
    i = np.asarray(i, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    max_abs = float(max(np.max(np.abs(i)), np.max(np.abs(q)), 1e-6))

    circles = []
    for i_value, q_value in zip(i, q):
        px = width / 2 + float(i_value) / max_abs * width * 0.42
        py = height / 2 - float(q_value) / max_abs * height * 0.42
        circles.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.2" fill="#2563eb" opacity="0.65" />')
    return "\n".join(circles)


def save_debug_html(x, y, out_path: Path) -> None:
    width = 360
    height = 180
    sections = []

    for label, mod_name in LABEL_TO_MOD.items():
        idx = int(np.where(y == label)[0][0])
        iq = x[idx]
        i = iq[0]
        q = iq[1]

        i_line = points_to_polyline(i, width, height)
        q_line = points_to_polyline(q, width, height)
        dots = constellation_points(i, q, width, height)

        sections.append(
            f"""
            <section>
              <h2>{mod_name}</h2>
              <div class="row">
                <figure>
                  <figcaption>Waveform: first 64 IQ samples</figcaption>
                  <svg viewBox="0 0 {width} {height}">
                    <rect width="{width}" height="{height}" fill="#ffffff" />
                    <polyline points="{i_line}" fill="none" stroke="#dc2626" stroke-width="1.5" />
                    <polyline points="{q_line}" fill="none" stroke="#16a34a" stroke-width="1.5" />
                  </svg>
                  <p><span class="red">I</span> / <span class="green">Q</span></p>
                </figure>
                <figure>
                  <figcaption>Constellation: one noisy sequence</figcaption>
                  <svg viewBox="0 0 {width} {height}">
                    <rect width="{width}" height="{height}" fill="#ffffff" />
                    <line x1="{width / 2}" y1="0" x2="{width / 2}" y2="{height}" stroke="#d4d4d8" />
                    <line x1="0" y1="{height / 2}" x2="{width}" y2="{height / 2}" stroke="#d4d4d8" />
                    {dots}
                  </svg>
                </figure>
              </div>
            </section>
            """
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>IQ debug plot</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #18181b; }}
    h1 {{ font-size: 22px; }}
    h2 {{ font-size: 18px; margin-top: 26px; }}
    .row {{ display: flex; gap: 18px; flex-wrap: wrap; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 13px; margin-bottom: 6px; color: #3f3f46; }}
    svg {{ width: {width}px; height: {height}px; border: 1px solid #d4d4d8; }}
    p {{ margin-top: 4px; font-size: 13px; }}
    .red {{ color: #dc2626; }}
    .green {{ color: #16a34a; }}
  </style>
</head>
<body>
  <h1>IQ Dataset Debug Plot</h1>
  {"".join(sections)}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nmatplotlib not found; saved HTML debug plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--phase-offset", action="store_true")
    parser.add_argument("--phase-max-deg", type=float, default=22.5)
    parser.add_argument("--cfo-offset", action="store_true")
    parser.add_argument("--cfo-max", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/iq_4mods_awgn.npz"))
    parser.add_argument("--plot", type=Path, default=Path("plots/iq_debug.png"))
    args = parser.parse_args()

    assert args.samples_per_class > 0
    assert args.length > 0
    assert args.snr_min < args.snr_max
    assert args.phase_max_deg >= 0.0
    assert args.cfo_max >= 0.0

    set_seed(args.seed)
    x, x_noisy_b, x_clean, y, snr, snr_b, phase_a, phase_b, cfo_a, cfo_b = generate_dataset(
        samples_per_class=args.samples_per_class,
        length=args.length,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        phase_offset=args.phase_offset,
        phase_max_deg=args.phase_max_deg,
        cfo_offset=args.cfo_offset,
        cfo_max=args.cfo_max,
    )

    print_dataset_report(
        x,
        y,
        snr,
        x_clean=x_clean,
        x_noisy_b=x_noisy_b,
        snr_b=snr_b,
        phase_a=phase_a,
        phase_b=phase_b,
        cfo_a=cfo_a,
        cfo_b=cfo_b,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x=x,
        x_noisy=x,
        x_noisy_a=x,
        x_noisy_b=x_noisy_b,
        x_clean=x_clean,
        y=y,
        snr=snr,
        snr_a=snr,
        snr_b=snr_b,
        phase_a=phase_a,
        phase_b=phase_b,
        cfo_a=cfo_a,
        cfo_b=cfo_b,
        mod_names=np.array(MOD_NAMES),
    )
    print(f"\nSaved dataset: {args.out}")

    save_debug_plot(x, y, args.plot)
    print("\nPASS: data generation sanity checks completed.")


if __name__ == "__main__":
    main()
