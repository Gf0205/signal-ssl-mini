# Experiment Notes

## Research Question

How should self-supervised multi-view objectives be designed for communication
IQ signals so that the learned representation is robust to channel/noise
perturbations while preserving modulation-discriminative structure for AMC?

The current focus is not SOTA performance. The goal is to build a minimal,
reproducible research loop and identify which physical perturbations are
compatible with which SSL objectives.

## Common Setup

- Task: automatic modulation classification
- Classes: BPSK, QPSK, 8PSK, 16QAM
- IQ shape: `[N, 2, 128]`
- Base channel: AWGN, SNR sampled from 0 to 20 dB
- Model: Tiny IQ Transformer
- Patch size: 8
- Hidden dimension: 64
- Transformer encoder layers: 2
- Attention heads: 4
- Downstream label ratio: 10% unless otherwise noted
- Metrics: Accuracy, Macro F1, training loss
- Main comparison: scratch classifier vs SSL-pretrained backbone

## Main Objectives Tested

### Masked Raw-IQ Reconstruction

The model reconstructs masked raw noisy IQ patches.

Result:

- This objective did not reliably improve AMC.
- At larger local scale, it was slightly worse than scratch.

Interpretation:

- Raw masked noisy-IQ reconstruction is weakly aligned with AMC.
- It may encourage local interpolation or noisy sample recovery rather than
  modulation-discriminative structure.

### Noisy-to-Clean Denoising

Input is noisy IQ and target is clean IQ.

Representative 20k result:

```text
Scratch:           acc=0.5111 +/- 0.0046 | macro_f1=0.5043 +/- 0.0047
Denoise pretrained acc=0.6317 +/- 0.0040 | macro_f1=0.6290 +/- 0.0050
Delta:             acc=+0.1206            | macro_f1=+0.1247
```

Interpretation:

- Denoising is strongly aligned with AMC because it removes AWGN while keeping
  modulation structure.
- However, clean targets are synthetic-oracle supervision and are not generally
  available in real received data.

### NoisyA-to-NoisyB Multi-View Reconstruction

Two independently corrupted views are generated from the same clean waveform:

```text
noisy_a = clean + noise_a
noisy_b = clean + noise_b
```

The SSL task is to reconstruct noisy_b patches from noisy_a.

Unified 20k result:

```text
Scratch:             acc ~= 0.50
NoisyA -> Clean:     acc ~= 0.608
NoisyA -> NoisyB:    acc ~= 0.611
```

Interpretation:

- Clean target is not necessary in this synthetic setting.
- Independent noisy views already provide a useful self-supervised signal.
- The model appears to learn structure stable across noise realizations rather
  than memorizing a clean oracle.

## GPU Scale-Up: 50k AWGN Multi-View

After migration to RTX 3090, the main NoisyA-to-NoisyB objective was scaled
from 20k to 50k samples while keeping the model fixed.

Setup:

- N = 50k
- label ratio = 10%
- SSL epochs = 20
- finetune epochs = 50
- seeds = 1, 2, 3
- model unchanged: hidden=64, layers=2, heads=4

Best-val summary:

```text
Scratch:       acc=0.6403 +/- 0.0419 | macro_f1=0.6382 +/- 0.0416
Noisy2Noisy:   acc=0.7851 +/- 0.0063 | macro_f1=0.7863 +/- 0.0064
Delta:         acc=+0.1448            | macro_f1=+0.1481
```

Final-epoch summary:

```text
Scratch:       acc=0.6378 +/- 0.0407 | macro_f1=0.6370 +/- 0.0407
Noisy2Noisy:   acc=0.7695 +/- 0.0078 | macro_f1=0.7703 +/- 0.0078
Delta:         acc=+0.1317            | macro_f1=+0.1333
```

Interpretation:

- The NoisyA-to-NoisyB benefit survives GPU scale-up.
- The pretrained model is much more stable across seeds.
- Scratch improves with longer finetuning, so part of the earlier gap was due
  to optimization speed. But even with 50 finetuning epochs, the pretrained
  model keeps a large final and best-val advantage.

## Phase Augmentation

Two views include independent constant phase offsets:

```text
view_a = clean * exp(j * phi_a) + noise_a
view_b = clean * exp(j * phi_b) + noise_b
```

### 20k Phase Strength Sweep

Phase max values:

```text
0 / 5 / 10 / 22.5 / 45 degrees
```

Main observation:

- 5 to 10 degrees gave slight gains over 0 degrees.
- 22.5 and 45 degrees reduced the SSL benefit.

Interpretation:

- Mild phase perturbation can help representation robustness at small scale.
- But phase is not a pure nuisance variable for PSK/QAM modulation.
- Strong phase invariance can remove or distort discriminative structure.

### 50k Phase10 Check

Setup:

- N = 50k
- seed = 1
- phase_max_deg = 10
- same label ratio, model, and training budget as AWGN-only scale-up

AWGN-only seed 1 best-val:

```text
Scratch:       acc=0.5839 | macro_f1=0.5822
Noisy2Noisy:   acc=0.7788 | macro_f1=0.7808
Delta:         acc=+0.1949 | macro_f1=+0.1986
```

Phase10 seed 1 best-val:

```text
Scratch:       acc=0.5841 | macro_f1=0.5819
Noisy2Noisy:   acc=0.7445 | macro_f1=0.7479
Delta:         acc=+0.1604 | macro_f1=+0.1659
```

Interpretation:

- The downstream scratch difficulty is comparable.
- Phase10 still helps over scratch, but is weaker than AWGN-only.
- The mild phase gain observed at 20k did not clearly survive 50k scale-up.

## CFO Preliminary Check

CFO was implemented as:

```text
x[n] * exp(j * 2*pi*cfo*n)
```

Tested values:

```text
cfo_max = 0 / 0.01 / 0.05
```

Main observation:

```text
cfo=0:
  delta acc ~= +0.0857

cfo=0.01:
  final delta acc ~= -0.0068
  best delta acc  ~= +0.0048

cfo=0.05:
  final delta acc ~= +0.0444
  best delta acc  ~= +0.0653
```

SSL reconstruction MSE increased sharply:

```text
cfo=0:    val MSE ~= 0.16
cfo=0.01: val MSE ~= 0.55
cfo=0.05: val MSE ~= 0.59
```

Interpretation:

- The tested CFO values are not mild for length-128 IQ sequences.
- Independent CFO views make raw pointwise cross-view reconstruction much more
  difficult and partly ill-posed.
- CFO may be better handled by representation-level consistency or contrastive
  objectives rather than raw IQ reconstruction.

## CFO Representation Consistency

To test whether CFO is better modeled at the representation level, a minimal
two-view consistency objective was added:

```text
same base waveform
  -> CFO/noise view A -> backbone -> z_a
  -> CFO/noise view B -> backbone -> z_b

loss = 1 - cosine(z_a, z_b)
```

### Pure Cosine Consistency

Observation:

- Validation cosine quickly approached 1.0.
- Per-dimension representation standard deviation collapsed to around 0.001.
- Downstream improvement was essentially absent.

Representative seed-1 result:

```text
Scratch best acc:      0.4500
Pure cosine best acc:  0.4517
Delta:                 +0.0017
```

Interpretation:

- Pure cosine consistency is not usable by itself.
- It can satisfy the objective through collapsed or near-collapsed
  representations.

### Cosine + Variance Regularization

To reduce collapse, a variance regularizer was added:

```text
loss = 1 - cosine(z_a, z_b) + lambda * variance_loss(z_a, z_b)
```

Multi-seed result on 20k CFO=0.01:

```text
Final:
Scratch:              acc=0.4499 +/- 0.0080 | macro_f1=0.4465 +/- 0.0134
Consistency+Var:      acc=0.4628 +/- 0.0145 | macro_f1=0.4547 +/- 0.0174
Delta:                acc=+0.0129            | macro_f1=+0.0082

Best-val:
Scratch:              acc=0.4506 +/- 0.0075 | macro_f1=0.4477 +/- 0.0068
Consistency+Var:      acc=0.4639 +/- 0.0119 | macro_f1=0.4642 +/- 0.0119
Delta:                acc=+0.0133            | macro_f1=+0.0165
```

Interpretation:

- Variance regularization mitigates collapse.
- The downstream gain becomes stable across seeds, but remains small.
- CFO representation-level consistency is more reasonable than raw
  reconstruction, but this minimal objective is still weak.

### Cosine + Variance + Projector

A projection head was tested:

```text
backbone -> h
projector -> z
loss acts on z
downstream finetuning uses h
```

This tests whether direct constraints on the backbone CLS representation harm
the downstream AMC space.

Multi-seed result on 20k CFO=0.01:

```text
Final:
Scratch:                 acc=0.4499 +/- 0.0080 | macro_f1=0.4465 +/- 0.0134
Consistency+Var+Proj:    acc=0.4511 +/- 0.0261 | macro_f1=0.4513 +/- 0.0244
Delta:                   acc=+0.0012            | macro_f1=+0.0048

Best-val:
Scratch:                 acc=0.4506 +/- 0.0075 | macro_f1=0.4477 +/- 0.0068
Consistency+Var+Proj:    acc=0.4467 +/- 0.0259 | macro_f1=0.4520 +/- 0.0264
Delta:                   acc=-0.0039            | macro_f1=+0.0044
```

Interpretation:

- The projector did not improve downstream transfer.
- It increased seed variance and may have solved the consistency task mostly in
  projection space without improving the backbone.
- This negative result suggests that CFO's difficulty is not simply caused by
  directly constraining the backbone CLS representation.

## Perturbation-Objective Summary

| Perturbation | SSL objective | Result | Current judgment |
| --- | --- | --- | --- |
| AWGN | Raw cross-view reconstruction | Strong, stable positive gain | Best current mainline; high objective-perturbation match |
| Constant phase | Raw cross-view reconstruction | Mild phase helped at 20k, but phase10 was weaker than AWGN-only at 50k | Bounded invariance; phase can also carry discriminative structure |
| CFO | Raw cross-view reconstruction | High MSE, unstable or weak downstream gain | Raw pointwise reconstruction is not natural for time-varying phase drift |
| CFO | Pure cosine consistency | Near collapse, almost no downstream gain | Not usable without anti-collapse terms |
| CFO | Cosine + variance | Stable but small positive gain | Anti-collapse works; objective remains limited |
| CFO | Cosine + variance + projector | No improvement and higher variance | Projector did not solve the core CFO objective mismatch |

## Method Evolution

```text
masked raw-IQ reconstruction
  -> denoising noisy-to-clean
  -> noisy-to-noisy multi-view reconstruction
  -> bounded phase perturbation
  -> CFO raw-reconstruction failure
  -> anti-collapse representation consistency
  -> projector negative result
```

## Current Conclusions

### 1. AWGN Multi-View Is the Strongest Current Mainline

Independent noisy views consistently improve AMC representation learning.
The effect appears at 20k and 50k scale and remains strong across seeds.
The pretrained model also shows much lower seed variance than scratch.

### 2. Phase Augmentation Is Not a Stable Free Gain

Mild phase perturbation can help at small scale, but this improvement did not
clearly survive 50k scale-up. Phase offset is partly a nuisance factor and
partly tied to AMC-discriminative structure.

### 3. CFO Raw Cross-View Reconstruction Is Unstable

CFO introduces time-varying phase drift. It changes the view relation more
strongly than AWGN or constant phase rotation and makes raw reconstruction a
less natural SSL objective.

### 4. CFO Representation Consistency Helps Only Modestly So Far

Pure cosine consistency collapses. Adding variance regularization prevents
collapse and yields a stable but small downstream gain. Adding a projector does
not improve transfer and increases variance.

This suggests that CFO probably needs a more structured objective than simple
cosine consistency, such as covariance decorrelation, predictor-based SSL, or
communication-aware features that model phase drift more explicitly.

## Working Thesis

More augmentation is not automatically better. For communication IQ signals,
the effectiveness of SSL depends on the match between:

1. the physical perturbation used to create views, and
2. the invariance or reconstruction behavior imposed by the SSL objective.

Current evidence suggests:

```text
AWGN             -> suitable for raw cross-view reconstruction
constant phase   -> useful only in controlled strength and not always beneficial
CFO              -> raw reconstruction is mismatched; simple representation
                    consistency is better but still weak
```

## Recommended Next Steps

Short term:

1. Keep AWGN NoisyA-to-NoisyB as the main baseline.
2. Do not continue broad phase/CFO sweeps immediately.
3. Do not keep optimizing projector variants without a clearer objective reason.
4. Before more runs, organize the current evidence into a compact result table
   and decide which claim the next experiment should test.

Research direction:

- Move from "does SSL help AMC?" to:

```text
Which communication perturbations should be reconstructed, and which should
be made invariant only at the representation level?
```

Possible next experiment families, after pausing to review:

- Confirm AWGN NoisyA-to-NoisyB at larger scale, if the goal is to strengthen
  the main positive result.
- Design a more structured CFO objective, if the goal is to study objective
  matching for time-varying phase drift.
- Add a second downstream robustness evaluation, if the goal is to test whether
  learned representations transfer beyond the same AMC setup.
