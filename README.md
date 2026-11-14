# 5G CSI Vision Transformer Localization

This repository implements the complete mini-project in explicit, testable
stages: DeepMIMO generation, leakage-resistant preprocessing, CNN/ViT/hybrid
training, and controlled multipath, noise, antenna-count, and subcarrier-count
studies.

## Step 1 — Raw DeepMIMO data

For each selected user, the script produces:

- complex CSI with shape `[user, BS antenna, OFDM subcarrier]`;
- the corresponding 3D DeepMIMO receiver position;
- the original receiver index and a validity mask;
- a JSON integrity report and three validation figures.

The default experiment uses `o1_28`, BS 3, 1,000 neighboring users, an 8×8 BS array, one UE antenna, 64 subcarriers, and at most 10 paths. See `configs/o1_28_small.yaml` for every explicit choice.

The production data configuration uses a 20 m × 20 m rectangle containing a
100 × 100 grid (10,000 users): `configs/o1_28_2d.yaml`.

### Environment

Python 3.11 or newer is required by DeepMIMO v4. On this machine, `virtualenv` is available even though Ubuntu's `python3-venv` package is not:

```bash
virtualenv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### Tests that do not require the scenario

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### Generate the first sample

The first invocation downloads the ray-tracing scenario and can therefore take time and disk space:

```bash
.venv/bin/python scripts/prepare_deepmimo.py --download
```

After the scenario exists locally, omit `--download`:

```bash
.venv/bin/python scripts/prepare_deepmimo.py
```

Generate the production 2D dataset without overwriting the smoke-test data:

```bash
.venv/bin/python scripts/prepare_deepmimo.py --config configs/o1_28_2d.yaml
```

If the official server answers `HTTP 429: Too Many Requests`, wait before
retrying; repeated attempts extend the rate-limit window. Alternatively,
download `o1_28` from the official DeepMIMO scenario page, extract it as
`deepmimo_scenarios/o1_28/`, and run the command without `--download`. Do not
use an old v2/v3 archive with this v4 pipeline.

Expected output directory:

```text
data/raw/o1_28_small/
├── csi_raw.npz
├── validation_report.json
├── user_positions.png
├── csi_amplitude_db.png
└── csi_phase.png
```

`csi_raw.npz` intentionally stores raw complex CSI. Converting it to amplitude/phase channels belongs to Step 2.

## Step 2 — Spatial split and model-ready features

Step 2 assigns complete x-coordinate columns to validation, training, guard,
and test zones. Feature and target normalization is fitted using training users
only. Run:

```bash
.venv/bin/python scripts/preprocess_csi.py
```

The output in `data/processed/o1_28_2d/` contains float32 features ordered as
`[log-amplitude dB, cos(phase), sin(phase)]`, raw XY positions, normalized XY
targets, split codes/indices, figures, and complete normalization metadata.

## Step 3 — Naïve and CNN baselines

The CNN training script memory-maps Step 2 features, computes the mean-position
baseline, selects the best CNN epoch on validation error, and evaluates the test
region once at the end:

```bash
.venv/bin/python scripts/train_cnn.py
```

Results are written to `outputs/cnn_baseline/` and include the checkpoint,
metrics, history, test predictions, training curves, and test-error CDF.

## Step 4 — CSI Vision Transformer

The compact ViT uses 8×8 CSI patches and separate learned positional
embeddings for the antenna and OFDM-subcarrier axes. It uses the same spatial
splits, target normalization, metrics, and validation-based model selection as
the CNN:

```bash
.venv/bin/python scripts/train_vit.py
```

Results are written to `outputs/vit_baseline/`.

## Extension — CNN-Transformer hybrid

The hybrid first extracts local CSI patterns with a three-stage convolutional
stem, then applies self-attention to the resulting 8×8 token grid:

```bash
.venv/bin/python scripts/train_hybrid.py
```

Results are written to `outputs/hybrid_baseline/`.

## Extension — Multipath richness

The controlled multipath study retains at most 1, 3, 5, or 10 propagation
paths while keeping positions, spatial splits, ViT architecture, seed, and
training settings identical. After training the configured variants, generate
the comparison with:

```bash
.venv/bin/python scripts/compare_multipath.py
```

## Extension — Robustness to CSI noise

The robustness experiment adds per-user complex AWGN before feature extraction,
reuses only clean-training normalization statistics, and evaluates the frozen
CNN, ViT, and hybrid checkpoints over repeated noise realizations:

```bash
.venv/bin/python scripts/evaluate_noise_robustness.py
```

## Extension — System parameters

This controlled study varies one CSI dimension at a time. The antenna study
uses centered 4x4, 4x8, and 8x8 physical subarrays (16, 32, and 64 antennas)
with 64 subcarriers. The frequency study keeps all 64 antennas and uniformly
selects 16, 32, or 64 subcarriers over the same 50 MHz bandwidth. All variants
reuse the same users, spatial split, ViT hyperparameters, and random seed.

Prepare and train the four new variants (the 64x64 baseline is reused):

```bash
for name in antennas16 antennas32 subcarriers16 subcarriers32; do
  .venv/bin/python scripts/preprocess_csi.py --config configs/preprocess_${name}.yaml
  .venv/bin/python scripts/train_vit.py --config configs/vit_${name}.yaml
done
.venv/bin/python scripts/compare_system_parameters.py
```

After all models are trained, generate the fair test comparison and combined
CDF with:

```bash
.venv/bin/python scripts/compare_models.py
```
