import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(command: list[str], command_log: Path) -> None:
    line = " ".join(command)
    print("\n>>> " + line, flush=True)
    with command_log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--samples-per-class", type=int, default=25000)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label-ratio", type=float, default=0.1)
    parser.add_argument("--ssl-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--ssl-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/scale_noisy2noisy"))
    parser.add_argument("--result-root", type=Path, default=Path("results/scale_noisy2noisy"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/scale_noisy2noisy"))
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--reuse-checkpoint", action="store_true")
    args = parser.parse_args()

    assert args.samples_per_class > 0
    assert args.length > 0
    assert args.snr_min < args.snr_max
    assert args.ssl_epochs > 0
    assert args.finetune_epochs > 0
    assert args.ssl_batch_size > 0
    assert args.batch_size > 0
    assert 0.0 < args.label_ratio <= 1.0

    run_id = args.run_id or datetime.now().strftime("n%Y%m%d_%H%M%S")
    total_samples = args.samples_per_class * 4
    data_path = args.data_dir / f"iq_4mods_awgn_views_n{total_samples}.npz"
    plot_path = args.result_root / run_id / "iq_debug.png"
    checkpoint_dir = args.checkpoint_root / run_id
    result_dir = args.result_root / run_id
    log_dir = args.log_root / run_id
    checkpoint_path = checkpoint_dir / f"noisy2noisy_backbone_seed{args.seed}.pt"
    result_csv = result_dir / "noisy2noisy_compare.csv"
    live_log = log_dir / "noisy2noisy_compare.txt"
    config_path = result_dir / "config.json"
    command_log = result_dir / "commands.txt"

    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "run_id": run_id,
            "total_samples": total_samples,
            "data_path": str(data_path),
            "checkpoint_path": str(checkpoint_path),
            "result_csv": str(result_csv),
            "live_log": str(live_log),
            "objective": "noisy_a_to_noisy_b",
            "model": {
                "patch_size": 8,
                "hidden_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
            },
        }
    )
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    command_log.write_text("", encoding="utf-8")

    python = sys.executable
    print("=== Noisy-to-noisy scale-up run ===")
    print(f"Run id: {run_id}")
    print(f"Total samples: {total_samples}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Config: {config_path}")

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
                "--seed",
                str(args.data_seed),
                "--out",
                str(data_path),
                "--plot",
                str(plot_path),
            ],
            command_log,
        )

    if args.reuse_checkpoint and checkpoint_path.exists():
        print(f"Reusing checkpoint: {checkpoint_path}", flush=True)
    else:
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
                str(args.seed),
                "--device",
                args.device,
                "--out",
                str(checkpoint_path),
            ],
            command_log,
        )

    run_command(
        [
            python,
            "compare_noisy2noisy_seeds.py",
            "--data",
            str(data_path),
            "--seeds",
            str(args.seed),
            "--label-ratio",
            str(args.label_ratio),
            "--epochs",
            str(args.finetune_epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-prefix",
            "noisy2noisy_backbone_seed",
            "--out",
            str(result_csv),
            "--log-file",
            str(live_log),
        ],
        command_log,
    )

    print("\nSaved config:", config_path)
    print("Saved command log:", command_log)
    print("Saved comparison CSV:", result_csv)
    print("Saved live log:", live_log)
    print("\nPASS: noisy-to-noisy scale-up run completed.")


if __name__ == "__main__":
    main()
