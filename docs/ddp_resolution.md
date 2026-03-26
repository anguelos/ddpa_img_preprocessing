# ddp_resolution

Module for estimating pixels-per-inch (PPI) of charter images and training
neural-network regressors on annotated resolution data.

## Resolution estimators

Two heuristic estimators are provided, both returning a `(ppi, confidence)` tuple.

| Class | Method | Confidence |
|---|---|---|
| `FixedSizeResolutionEstimator` | assumes largest dimension = 35 cm | 0.20 |
| `CalibrationCardResolutionEstimator` | reads `.layout.pred.json`, finds `Img:CalibrationCard` bounding boxes and derives PPI from known physical dimensions (19.5 cm × 6.0 cm); falls back to fixed-size when no card is detected | 0.85 |

Run from the command line with [`ddp_res_offline`](cli.md).

---

## ResDs — training dataset

`ResDs` is a PyTorch `Dataset` for PPI regression.  Each sample is an image
region (full image or a dominant crop) paired with a ground-truth PPI value
derived from ruler annotations stored in `.resolution.gt.json` files.

### Construction

```python
from ddp_resolution.res_ds import ResDs

# From an explicit list of annotation paths
ds = ResDs(gt_paths, image_crop="img")

# Convenience: glob under a root directory
ds = ResDs.from_root(
    data_root_path="/data/charters",
    resolution_gt_glob="**/*.resolution.gt.json",
    image_crop="Img:WritableArea",   # or "Wr:OldText" or "img"
    dominant_area_ratio=0.5,         # optional override
)
```

### Key class constants (overridable in subclasses)

| Constant | Default | Meaning |
|---|---|---|
| `KNOWN_SIZES_CM` | `{"5cm": 5.0, "1cm": 1.0, "5in": 12.7, "1in": 2.54}` | Physical sizes of supported ruler annotations |
| `DOMINANT_AREA_RATIO` | `0.5` | A crop class rect must cover strictly more than this fraction of total class area to be considered dominant |
| `CROP_CLASSES` | `("Img:WritableArea", "Wr:OldText", "img")` | Accepted values for `image_crop` |

`dominant_area_ratio` can also be set per-instance via the constructor parameter.

### Transforms

Factory functions for torchvision pipelines (ImageNet normalisation applied in both):

```python
from ddp_resolution.res_ds import make_train_transform, make_inference_transform

# Training: random flips, colour jitter, optional random crop
train_t = make_train_transform(patch_size=224, jitter_strength=0.3)

# Inference: optional centre crop, no augmentation
infer_t = make_inference_transform(patch_size=224)

ds.input_transform = train_t
```

If `patch_size` is omitted the spatial size of the image is preserved.
`pad_if_needed=True` is set on the training random crop so images smaller
than `patch_size` are padded rather than crashing.

### Splitting

```python
train_ds, val_ds = ds.random_split(ratio=0.8, seed=42)
```

`_view` is used internally so both halves share the same class configuration
and `dominant_area_ratio` without re-scanning the filesystem.

---

## Evaluation

### `ResDs.evaluate(pred_ext=".res.pred.json")`

For each sample, looks for a prediction JSON next to the image (replacing
`.img.<ext>` with `pred_ext`).  The prediction file is the output of
[`ddp_res_offline`](cli.md) and must contain a
`"ppi"` key.

Returns a dict:

| Key | Description |
|---|---|
| `n_total` | Total samples in this dataset view |
| `n_predicted` | Samples for which a readable prediction was found |
| `coverage` | `n_predicted / n_total` |
| `mean_gt_ppi` | Mean GT PPI across **all** samples — use to contextualise `mae` |
| `mae` | Mean absolute error (PPI) |
| `mape` | Mean absolute percentage error (0–100 scale) |
| `rmse` | Root-mean-squared error (PPI) |
| `median_ae` | Median absolute error (PPI) |
| `log2_mae` | Mean \|log₂(pred/gt)\|; 1.0 = off by one doubling on average |
| `log2_rmse` | RMS of log₂(pred/gt) errors |

All error metrics are `None` when `n_predicted == 0`.

`log2_mae` and `log2_rmse` are the primary scale-invariant metrics: a 2× over-
estimate and a 0.5× under-estimate are both `log2_mae = 1.0`, whereas MAPE
would report 100 % and 50 % respectively for the same relative mistake.

### `ddp_res_evaluate` CLI

```
ddp_res_evaluate -fsdb_roots <dir> [<dir> ...] [-gt_glob GLOB]
                 [-image_crop img|Img:WritableArea|Wr:OldText]
                 [-pred_exts EXT[,EXT ...]]
                 [-plot] [-plotfname FILENAME]
                 [-verbose]
```

Prints a table with one row per prediction-file extension and all metrics as
columns:

```
ext                  n_total  n_predicted     coverage  mean_gt_ppi          mae  ...
------------------------------------------------------------------------------------
.res.pred.json           523          498        0.952      312.400       28.700  ...
.fixed_size.json         523          523        1.000      312.400      104.200  ...
```

With `-plot` a scatter plot of predicted vs GT PPI is shown interactively
(`plt.show()`); pass `-plotfname path/to/out.png` to save to a file instead.
Multiple extensions are distinguished by colour with a legend; a grey dashed
diagonal marks the ideal prediction.

Example:

```bash
ddp_res_evaluate \
    -fsdb_roots /data/charters \
    -pred_exts .res.pred.json,.fixed_size.json \
    -plot -plotfname eval.png -verbose
```

### `end to end usage` CLI

Download the dataset from google drive (in this no partition is done)
Run ddp_res_offline methods.
Evaluate their performance.

```
pip install gdown
mkdir -p sample_data/1000_CVCharters_with_res

cd sample_data && gdown 12eaDPctQxZlrC0DE1R8GyuBoutzqBQe8 
cd 1000_CVCharters_with_res
tar -xpvzf ../1000_CVCharters_with_res.tar.gz
cd ../..

ddp_res_offline -method layoutgt -images ./sample_data/1000_CVCharters_with_res/*/*/*/*.img.* -verbose
ddp_res_offline -method layout -images ./sample_data/1000_CVCharters_with_res/*/*/*/*.img.* -verbose
ddp_res_offline -method fixed_size -images ./sample_data/1000_CVCharters_with_res/*/*/*/*.img.* -verbose

ddp_res_evaluate -fsdb_root ./sample_data/1000_CVCharters_with_res/ -pred_exts '.res.pred_layout.json' '.res.pred_layoutgt.json' '.res.pred_fixed.json' -verbose 

# Or run with -plot for an plot
#ddp_res_evaluate -fsdb_root ./sample_data/1000_CVCharters_with_res/ -pred_exts '.res.pred_layout.json' '.res.pred_layoutgt.json' '.res.pred_fixed.json' -verbose -plot
```