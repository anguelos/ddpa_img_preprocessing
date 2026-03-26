# DDPA Image Preprocessing

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Read the Docs](https://img.shields.io/readthedocs/ddpa-img-preprocessing)](https://ddpa-img-preprocessing.readthedocs.io)
[![GitHub repo size](https://img.shields.io/github/repo-size/anguelos/ddpa_img_preprocessing)](https://github.com/anguelos/ddpa_img_preprocessing)
[![GitHub last commit](https://img.shields.io/github/last-commit/anguelos/ddpa_img_preprocessing)](https://github.com/anguelos/ddpa_img_preprocessing/commits/main)
[![PyPI downloads](https://img.shields.io/pypi/dm/ddpa-img-preprocessing)](https://pypi.org/project/ddpa-img-preprocessing/)
[![Tests](https://img.shields.io/github/actions/workflow/status/anguelos/ddpa_img_preprocessing/tests.yml?label=tests)](https://github.com/anguelos/ddpa_img_preprocessing/actions)

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
