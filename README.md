# signal-ssl-mini

Minimal PyTorch experiments for self-supervised IQ signal pretraining and AMC
classification.

## Scope

- Four modulation classes: BPSK, QPSK, 8PSK, 16QAM
- Synthetic IQ data with shape `[N, 2, 128]`
- Tiny IQ Transformer backbone
- Scratch AMC baseline
- Masked reconstruction SSL
- Denoising pretraining
- NoisyA-to-NoisyB multi-view pretraining
- Phase and CFO augmentation probes

This repository intentionally stores code only. Generated datasets,
checkpoints, plots, logs, and result CSV files are excluded by `.gitignore`.

## v0.1 Research Baseline

The current research focus is no longer just whether SSL helps AMC. The
emerging question is how communication perturbations should be matched with
self-supervised objectives.

Current stage conclusions:

- AWGN: raw noisy-view reconstruction is the strongest and most stable
  objective tested so far.
- Constant phase offset: useful only within a bounded strength; excessive phase
  invariance can weaken AMC-discriminative structure.
- CFO: raw reconstruction is mismatched, while simple representation
  consistency avoids collapse only after variance regularization and gives
  small gains.

See `experiment_notes.md` for the full perturbation-objective summary table and
the method evolution log. The next recommended stage is external validity:
public AMC data or a more realistic signal chain, rather than adding more model
tricks.

## Quick GPU Smoke Test

```bash
python data.py --samples-per-class 16 --length 128 --out data/smoke_gpu_views.npz --plot plots/smoke_gpu.png
python gpu_smoke.py --data data/smoke_gpu_views.npz --batch-size 32 --device cuda
```

Expected pass signal:

```text
CUDA available: True
Classifier batch device: x=cuda:0
Noisy pair batch device: a=cuda:0
PASS: GPU smoke test completed.
```

## Core Noisy-to-Noisy Check

```bash
python data.py --samples-per-class 5000 --length 128 --out data/iq_4mods_awgn_views_n20000.npz --plot plots/iq_debug.png
python pretrain_noisy2noisy.py --data data/iq_4mods_awgn_views_n20000.npz --epochs 10 --seed 1 --device cuda --out checkpoints/noisy2noisy_backbone_seed1.pt
python compare_noisy2noisy_seeds.py --data data/iq_4mods_awgn_views_n20000.npz --seeds 1 --label-ratio 0.1 --epochs 10 --device cuda --checkpoint-dir checkpoints --checkpoint-prefix noisy2noisy_backbone_seed --out results/noisy2noisy_seed1.csv --log-file logs/noisy2noisy_seed1.txt
```

## First GPU Scale-Up

This keeps the model fixed and only increases the synthetic dataset size.

```bash
python run_scale_noisy2noisy.py --run-id n100k_seed1_e30 --samples-per-class 25000 --seed 1 --label-ratio 0.1 --ssl-epochs 30 --finetune-epochs 30 --ssl-batch-size 256 --batch-size 256 --device cuda
```

## Public Dataset First Check

For the first external-validity step, use RadioML2016.10A rather than the much
larger RadioML2018.01A. RadioML2016.10A already uses `[N, 2, 128]` IQ samples,
so it matches the current Tiny IQ Transformer setup.

Download the official `RML2016.10a_dict.pkl` file manually and place it under
`data/`. Do not commit it to Git.

Convert the four-class subset:

```bash
python prepare_radioml2016.py \
  --input data/RML2016.10a_dict.pkl \
  --out data/radioml2016_4mods_views.npz \
  --normalize \
  --make-views \
  --view-snr-min 10 \
  --view-snr-max 20
```

Then run the normal dataset sanity check:

```bash
python dataset.py --data data/radioml2016_4mods_views.npz --batch-size 64
```

The converted file contains:

- `x`, `y`, `snr` for downstream AMC.
- `x_noisy_a`, `x_noisy_b`, `snr_a`, `snr_b` for reusing the NoisyA-to-NoisyB
  pretraining script.

Note: RadioML2016.10A is distributed as a Python pickle. Only load pickle files
from a trusted source.
