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
ddp_binarize_offline img_abc123.img.jpg img_def456.img.jpg

# Estimate resolution
ddp_resolution_offline img_abc123.img.jpg

# Select recto
ddp_recto_offline /path/to/charter/dir

# Full offline preprocessing of an FSDB
ddp_cv_preprocess_offline /path/to/fsdb_root
```

## FSDB Structure

```
fsdb_root/
  ARCHIVENAME/
    <fond_md5>/
      <charter_md5>/
        CH.cei.xml
        CH.url.txt
        CH.atom_id.txt
        image_urls.json
        <img_md5>.img.<ext>
```

## License

AGPL-3.0 — see LICENSE file.
