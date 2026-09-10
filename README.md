# Decathlon Lung Tumor Segmentation

This repository is being upgraded from a historical 2D prototype into a
reproducible **3D CT** lung-tumor segmentation research framework. The primary
development dataset is Medical Segmentation Decathlon (MSD) Task06 Lung using
the original NIfTI volumes.

> The 3D pipeline is under development. No novelty, superiority, or
> state-of-the-art claim is made, and no new experimental results are available
> yet.

## Project status

**WP1 — Foundation is complete.** The source package now also contains the WP2
data audit/split implementation, WP3 core transforms, WP4 models/losses/metrics,
and WP5 training/inference primitives. These components pass synthetic 3D NIfTI
tests, but real-data acceptance is intentionally pending until the original MSD
Task06 Lung dataset is available. The remaining validation gates are listed in
the current-limitations and planned-workflow sections below.

## Implemented 3D source code

The maintained application is under `src/lungseg`; the archived 2D files are not
imported by it.

```text
src/lungseg/
├── app.py                  # unified main.py command implementations
├── data/
│   ├── discovery.py        # deterministic image/label pairing
│   ├── audit.py            # geometry, intensity, and tumor statistics
│   ├── splits.py           # leakage-safe patient-level folds
│   ├── transforms.py       # MONAI 3D preprocessing and augmentation
│   └── dataset.py          # volumetric datasets and loaders
├── models/
│   ├── blocks.py           # 3D residual blocks and attention gates
│   ├── ra_segnet3d.py      # custom residual-attention encoder-decoder
│   └── registry.py         # 3D U-Net, SegResNet, RA-SegNet3D
├── losses/segmentation.py  # Dice + cross-entropy
├── metrics/segmentation.py # case Dice, HD95, IoU, precision, recall
├── engine/
│   ├── trainer.py          # training and whole-volume validation primitives
│   └── inference.py        # sliding-window prediction and NIfTI output
└── utils/                  # configuration, provenance, JSON, seeding
```

The custom `RA-SegNet3D` has explicit multi-scale skips, residual encoder and
decoder blocks, and trainable 3D attention gates. It is an experimental model,
not a validated proposed method or a performance claim.

## Legacy 2D implementation

The original slice-based SegNet-like implementation is preserved in
[`legacy/2d_segnet`](legacy/2d_segnet). Commit `98d7d11` is tagged locally as
`legacy-2d-v1`. Its reported results are historical and have not been reproduced
under the planned patient-level, five-fold 3D evaluation protocol.

The legacy README incorrectly called Task06 Lung an MRI dataset. It is a CT
dataset. The original file is retained unmodified for provenance, while the
legacy notes explain its known limitations.

## Installation

Python 3.10 or newer is required. Create an isolated environment and install the
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build appropriate for the host before the
editable installation when GPU training is required. nnU-Net v2 will be managed
separately when its canonical baseline is added.

## Dataset setup

Obtain the original MSD Task06 Lung package and leave its NIfTI files unchanged:

```text
data/raw/Task06_Lung/
├── imagesTr/
├── labelsTr/
├── imagesTs/
└── dataset.json
```

The `data/` directory and NIfTI files are ignored by Git. Do not commit patient
data or generated medical images. Dataset paths are supplied through
configuration; no private absolute path belongs in source code.

Spacing, CT intensity clipping, normalization statistics, and patch dimensions
are intentionally marked `null` in the initial configuration. They will be
selected from training-fold audits rather than guessed or derived from
validation/test cases.

## Audit and cross-validation splits

All project operations use the root `main.py`. Display available commands with:

```bash
python main.py --help
python main.py status
```

After placing the raw dataset at the configured location, audit every labeled case:

```bash
python main.py inspect \
  --data data/raw/Task06_Lung \
  --output artifacts/audits/msd_lung
```

The command writes `cases.json`, `cases.csv`, and `summary.json`. It fails on
missing labels, shape mismatches, affine mismatches, non-finite intensities, and
invalid labels instead of silently dropping patients.

Generate deterministic patient-level folds with:

```bash
python main.py split \
  --data data/raw/Task06_Lung \
  --output artifacts/splits/msd_lung_5fold_seed43.json \
  --folds 5 \
  --seed 43
```

All labeled cases appear in validation exactly once. Tumor-volume
stratification is used only when each resulting bin has enough cases; otherwise
the command records a seeded K-fold fallback.

## Configuration

Initial configuration files are grouped by purpose:

```text
configs/data/msd_lung.yaml
configs/model/{unet3d,segresnet,ra_segnet3d}.yaml
configs/experiment/{smoke,baseline_unet3d}.yaml
```

Load and override them with the small configuration utility:

```python
from pathlib import Path
from lungseg.config import load_yaml, merge_configs

data = load_yaml(Path("configs/data/msd_lung.yaml"))
experiment = load_yaml(Path("configs/experiment/smoke.yaml"))
config = merge_configs(data, experiment)
```

## Reproducibility report

Capture software, hardware, Git commit, and working-tree state with:

```bash
python main.py environment --output artifacts/environment.json
```

## Training, evaluation, and inference

After audit-supported spacing, HU window, and patch size have been written to
`configs/data/msd_lung.yaml`, run one configured fold with:

```bash
python main.py train \
  --data-config configs/data/msd_lung.yaml \
  --model-config configs/model/unet3d.yaml \
  --experiment-config configs/experiment/smoke.yaml \
  --split artifacts/splits/msd_lung_5fold_seed43.json
```

`python main.py run` is an alias for `train`. Each run stores its resolved
configuration, environment, split, history, case metrics, and best/last
checkpoints under `artifacts/runs/`.

Evaluate a checkpoint:

```bash
python main.py evaluate \
  --checkpoint artifacts/runs/RUN/checkpoints/best.pt \
  --split artifacts/splits/msd_lung_5fold_seed43.json \
  --fold 0
```

Infer one NIfTI volume and restore its prediction to original geometry:

```bash
python main.py infer \
  --checkpoint artifacts/runs/RUN/checkpoints/best.pt \
  --image data/raw/Task06_Lung/imagesTr/lung_001.nii.gz \
  --output artifacts/predictions/lung_001.nii.gz
```

## Tests

Tests use generated NIfTI fixtures and no patient data:

```bash
python main.py test
python main.py test --quiet
```

The suite covers data discovery/audit, fold integrity, paired 3D transforms,
model output and gradient flow, Dice/HD95 edge cases, a CPU optimizer step,
sliding-window inference, and affine-preserving NIfTI output.

## Current limitations

- Real Task06 Lung audit parameters have not been selected.
- Whole-volume preprocessing inversion must be completed and visually verified
  before predictions from resampled/cropped data are treated as final outputs.
- No full fold has been trained and no quantitative model result is verified.
- nnU-Net v2, Attention U-Net, and MedNeXt baselines are not integrated yet.
- External validation remains intentionally deferred.

## Planned research workflow

```text
Original Task06 Lung NIfTI
        ↓
dataset audit and deterministic patient-level folds
        ↓
traceable 3D preprocessing and patch sampling
        ↓
3D U-Net baseline
        ↓
sliding-window whole-volume inference
        ↓
per-patient Dice and HD95
        ↓
residual-attention model and controlled ablations
```

## Results

There are currently no verified 3D research results. Results will be added only
after case-level evaluation artifacts and their exact configurations are
available.

## License

This project is distributed under the GNU General Public License v3.0. See
[`LICENSE`](LICENSE).
