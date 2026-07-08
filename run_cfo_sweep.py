import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


def cfo_tag(cfo_max: float) -> str:
    if abs(cfo_max) < 1e-12:
        return "000"
    return str(cfo_max).replace(".", "p").replace("-", "m")


def label_tag(label_ratio: float) -> str:
    return f"{int(round(label_ratio * 1000)):03d}"


def run_command(command: list[str]) -> None:
    print("\n>>> " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.array(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def summarize_cfo_csv(csv_path: Path, cfo_max: float) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing comparison CSV: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    summary_rows = []
    for method in ["scratch", "noisy2noisy_pretrained"]:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            raise ValueError(f"No rows for method={method} in {csv_path}")

        final_acc_mean, final_acc_std = mean_std(
            [float(row["test_acc"]) for row in method_rows]
        )
        final_f1_mean, final_f1_std = mean_std(
            [float(row["test_macro_f1"]) for row in method_rows]
        )
        best_acc_mean, best_acc_std = mean_std(
            [float(row["best_test_acc"]) for row in method_rows]
        )
        best_f1_mean, best_f1_std = mean_std(
            [float(row["best_test_macro_f1"]) for row in method_rows]
        )

        summary_rows.append(
            {
                "cfo_max": cfo_max,
                "method": method,
                "final_acc_mean": final_acc_mean,
                "final_acc_std": final_acc_std,
                "final_macro_f1_mean": final_f1_mean,
                "final_macro_f1_std": final_f1_std,
                "best_acc_mean": best_acc_mean,
                "best_acc_std": best_acc_std,
                "best_macro_f1_mean": best_f1_mean,
                "best_macro_f1_std": best_f1_std,
            }
        )

    by_method = {row["method"]: row for row in summary_rows}
    scratch = by_method["scratch"]
    ssl = by_method["noisy2noisy_pretrained"]
    summary_rows.append(
        {
            "cfo_max": cfo_max,
            "method": "delta_noisy2noisy_minus_scratch",
            "final_acc_mean": ssl["final_acc_mean"] - scratch["final_acc_mean"],
            "final_acc_std": "",
            "final_macro_f1_mean": ssl["final_macro_f1_mean"]
            - scratch["final_macro_f1_mean"],
            "final_macro_f1_std": "",
            "best_acc_mean": ssl["best_acc_mean"] - scratch["best_acc_mean"],
            "best_acc_std": "",
            "best_macro_f1_mean": ssl["best_macro_f1_mean"]
            - scratch["best_macro_f1_mean"],
            "best_macro_f1_std": "",
        }
    )
    return summary_rows


def write_summary(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cfo_max",
        "method",
        "final_acc_mean",
        "final_acc_std",
        "final_macro_f1_mean",
        "final_macro_f1_std",
        "best_acc_mean",
        "best_acc_std",
        "best_macro_f1_mean",
        "best_macro_f1_std",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_compact_summary(rows: list[dict]) -> None:
    print("\n=== CFO strength sweep summary ===")
    for row in rows:
        if row["method"] != "delta_noisy2noisy_minus_scratch":
            continue
        print(
            f"cfo_max={row['cfo_max']:>5} | "
            f"delta final acc={row['final_acc_mean']:+.4f} | "
            f"delta final f1={row['final_macro_f1_mean']:+.4f} | "
            f"delta best acc={row['best_acc_mean']:+.4f} | "
            f"delta best f1={row['best_macro_f1_mean']:+.4f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfos", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    parser.add_argument("--samples-per-class", type=int, default=5000)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--ssl-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--label-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ssl-batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/cfo_sweep"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/cfo_sweep"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/cfo_sweep"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/cfo_sweep_summary.csv"))
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    args = parser.parse_args()

    assert len(args.cfos) > 0
    assert args.samples_per_class > 0
    assert args.length > 0
    assert args.snr_min < args.snr_max
    assert len(args.seeds) > 0
    assert args.ssl_epochs > 0
    assert args.finetune_epochs > 0
    assert 0.0 < args.label_ratio <= 1.0
    assert all(cfo >= 0.0 for cfo in args.cfos)

    python = sys.executable
    all_summary_rows = []

    print("=== CFO strength sweep setup ===")
    print(f"Python: {python}")
    print(f"Normalized CFO max values: {args.cfos}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Seeds: {args.seeds}")
    print(f"SSL epochs: {args.ssl_epochs}")
    print(f"Finetune epochs: {args.finetune_epochs}")
    print(f"Label ratio: {args.label_ratio}")
    print("CFO definition: x[n] * exp(j * 2*pi*cfo*n)")

    for cfo_max in args.cfos:
        tag = cfo_tag(cfo_max)
        total_samples = args.samples_per_class * 4
        data_path = args.data_dir / f"iq_4mods_awgn_cfo{tag}_views_n{total_samples}.npz"
        plot_path = args.results_dir / f"cfo{tag}_debug.png"
        cfo_ckpt_dir = args.checkpoint_dir / f"cfo{tag}"
        result_csv = args.results_dir / f"cfo{tag}_label{label_tag(args.label_ratio)}.csv"
        log_file = args.logs_dir / f"cfo{tag}_label{label_tag(args.label_ratio)}.txt"

        print(f"\n=== Normalized CFO max: {cfo_max} ===", flush=True)

        if args.reuse_data and data_path.exists():
            print(f"Reusing dataset: {data_path}", flush=True)
        else:
            run_command(
                [
                    python,
                    "data.py",
                    "--samples-per-class",
                    str(args.samples_per_class),
                    "--length",
                    str(args.length),
                    "--snr-min",
                    str(args.snr_min),
                    "--snr-max",
                    str(args.snr_max),
                    "--cfo-offset",
                    "--cfo-max",
                    str(cfo_max),
                    "--seed",
                    str(args.data_seed),
                    "--out",
                    str(data_path),
                    "--plot",
                    str(plot_path),
                ]
            )

        cfo_ckpt_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            ckpt_path = cfo_ckpt_dir / f"noisy2noisy_backbone_seed{seed}.pt"
            if args.reuse_checkpoints and ckpt_path.exists():
                print(f"Reusing checkpoint: {ckpt_path}", flush=True)
                continue

            run_command(
                [
                    python,
                    "pretrain_noisy2noisy.py",
                    "--data",
                    str(data_path),
                    "--epochs",
                    str(args.ssl_epochs),
                    "--batch-size",
                    str(args.ssl_batch_size),
                    "--seed",
                    str(seed),
                    "--out",
                    str(ckpt_path),
                ]
            )

        run_command(
            [
                python,
                "compare_noisy2noisy_seeds.py",
                "--data",
                str(data_path),
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--label-ratio",
                str(args.label_ratio),
                "--epochs",
                str(args.finetune_epochs),
                "--batch-size",
                str(args.batch_size),
                "--checkpoint-dir",
                str(cfo_ckpt_dir),
                "--checkpoint-prefix",
                "noisy2noisy_backbone_seed",
                "--out",
                str(result_csv),
                "--log-file",
                str(log_file),
            ]
        )

        all_summary_rows.extend(summarize_cfo_csv(result_csv, cfo_max))
        write_summary(all_summary_rows, args.summary_out)
        print_compact_summary(all_summary_rows)
        print(f"Saved running summary CSV: {args.summary_out}", flush=True)

    print("\nPASS: CFO strength sweep completed.")


if __name__ == "__main__":
    main()
