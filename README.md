# DDPA Image Preprocessing

Preprocessing pipeline for medieval charter images stored in a Filesystem Database (FSDB).

## Overview

This package implements three preprocessing tasks for charter images:

1. **Binarization** (`ddp_binarize`) — Converts images to grayscale semantic binarization (0=text, 255=background)
2. **Resolution Estimation** (`ddp_resolution`) — Estimates pixels-per-inch (PPI) of charter images
3. **Recto Selection** (`ddp_recto`) — Selects the most appropriate image for text analysis

## Installation

```bash
pip install -e ".[dev]"
```

Or with torch_mentor (required for ML models):
```bash
pip install -e ../torch_mentor
pip install -e ".[dev]"
```

## Quick Start

```bash
# Binarize images
ddp_binarize_offline -images ./test/fake_fsdb_root/*/*/*/*.img.* -method otsu -verbose

# Estimate resolution (calibration card heuristic)
ddp_resolution_offline -images ./test/fake_fsdb_root/*/*/*/*.img.* -method layout -verbose

# Estimate resolution (fixed size heuristic)
ddp_resolution_offline -images ./test/fake_fsdb_root/*/*/*/*.img.* -method fixed_size -verbose

# Select recto for charter directories
ddp_recto_offline -charter_dir ./test/fake_fsdb_root/*/*/* -verbose

# Full offline preprocessing of an FSDB
ddp_cv_preprocess_offline -fsdb_root ./test/fake_fsdb_root -verbose
```

## License

AGPL-3.0 — see LICENSE file.
