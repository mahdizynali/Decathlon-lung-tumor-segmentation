# Legacy 2D segmentation prototype

This directory preserves the repository state represented by Git tag
`legacy-2d-v1` (commit `98d7d11`) while new development proceeds in `src/`.

## Historical behavior

The prototype:

- converts NIfTI CT volumes into independent 2D NumPy slices;
- resizes slices to `224 × 224`;
- optionally duplicates grayscale CT into three channels;
- drops the first 30 slices by default;
- accepts only one hardcoded affine orientation;
- divides intensities by the fixed value `3071`;
- writes JPG previews while loading training samples;
- uses a SegNet-like residual/channel-attention model and PyTorch Lightning.

These choices are retained only for provenance and are not the protocol for new
research. In particular, the architecture is not a classical pooling-index
SegNet and its slice-level workflow does not preserve volumetric context.

## Reproduction limitations

The historical repository did not include a pinned dependency manifest. Its
PyTorch Lightning and TorchMetrics APIs are incompatible with some modern
versions. `datasets/preprocessing.py` also runs preprocessing at import time and
contains historical relative paths. Run legacy code only in an isolated
environment after reviewing those paths and behaviors.

`README.original.md` is the unmodified historical README. Its MRI description
is incorrect—MSD Task06 Lung contains CT—and its metrics have not been verified
under patient-level cross-validation. They must not be compared directly with
future 3D results.
