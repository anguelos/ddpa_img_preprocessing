# Quickstart

## Installation

```bash
git clone <repo>
cd ddpa_img_preprocessing
pip install -e ".[dev]"
```

## Running the offline pipeline

```bash
ddp_cv_preprocess_offline /path/to/your/fsdb_root
```

This processes all charters in the FSDB, computing:
- Binarization (`<img_md5>.bin.png`)
- Resolution estimate (`<img_md5>.resolution.json`)
- Recto symlink (`CH.recto.<ext>`)

## Individual tools

### Binarize a single image

```bash
ddp_binarize_offline myimage.img.jpg
# produces myimage.bin.png
```

### Estimate resolution

```bash
ddp_resolution_offline myimage.img.jpg
# produces myimage.resolution.json
```

### Select recto for a charter

```bash
ddp_recto_offline /path/to/charter/directory
```

## Running tests

```bash
make test        # stop on first failure
make testfull    # run all tests
make unitest     # unittests with coverage
```
