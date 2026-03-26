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

This processes all charters in the FSDB (see {doc}`fsdb`) computing:
- Binarization (`<img_md5>.bin.png`)
- Resolution estimate (`<img_md5>.resolution.json`)
- Recto symlink (`CH.recto.<ext>`)

## Individual tools

### Binarize a single image

```bash
ddp_binarize_offline -images ./test/fake_fsdb_root/*/*/*/*.img.* -method otsu -verbose
# produces <img_md5>.bin.png alongside each input image
```

### Estimate resolution

```bash
ddp_res_offline -images ./test/fake_fsdb_root/*/*/*/*.img.* -method layout -verbose
# produces <img_md5>.resolution.json alongside each input image
```

### Select recto for a charter

```bash
ddp_recto_offline -charter_dir ./test/fake_fsdb_root/*/*/* -verbose
```

## Running tests

```bash
make test        # stop on first failure
make testfull    # run all tests
make unitest     # unittests with coverage
```
