# Command Line Tools

## ddp_binarize_offline

Binarizes charter images. Output is grayscale where 0=foreground (text), 255=background.

```
ddp_binarize_offline -images <image> [<image> ...] [-method otsu|bunet] [-verbose]
```

Example:
```bash
ddp_binarize_offline -images ./test/fake_fsdb_root/*/*/*/*.img.* -method otsu -verbose
```

Output file: `<img_md5>.bin.png` (same directory as input)

## ddp_res_offline

Estimates pixels-per-inch of charter images.

```
ddp_res_offline -images <image> [<image> ...] [-method layout|fixed_size|resresnet] [-verbose] [-save_resized] [-target_ppi 300.0] [-max_size 4096] [-min_size 256]
```

Example:
```bash
ddp_res_offline -images ./test/fake_fsdb_root/*/*/*/*.img.* -method layout -verbose
```

Output file: `<img_md5>.resolution.json`

## ddp_recto_offline

Selects the most appropriate image (recto) for each charter directory.

```
ddp_recto_offline -charter_dir <dir> [<dir> ...] [-nosymlinks] [-min_recto_prob 0.4] [-verbose]
```

Example:
```bash
ddp_recto_offline -charter_dir ./test/fake_fsdb_root/*/*/* -verbose
```

Creates symlink `CH.recto.<ext>` if best image probability >= min_recto_prob.

## ddp_cv_preprocess_offline

Runs the full preprocessing pipeline on one or more FSDB roots.

```
ddp_cv_preprocess_offline -fsdb_root <dir> [<dir> ...] [-nosymlinks] [-min_recto_prob 0.4] [-verbose]
```

Example:
```bash
ddp_cv_preprocess_offline -fsdb_root ./test/fake_fsdb_root -verbose
```

## ddp_res_evaluate

Evaluates resolution prediction files against ground-truth annotations in a ResDs root.

```
ddp_res_evaluate -fsdb_roots <dir> [<dir> ...] [-gt_glob GLOB]
                 [-image_crop img|Img:WritableArea|Wr:OldText]
                 [-pred_exts EXT[,EXT,...]] [-plot] [-plotfname FILENAME]
                 [-verbose]
```

Example:
```bash
ddp_res_evaluate -fsdb_roots ./sample_data/1000_CVCharters_with_res \
    -pred_exts .res.pred.json,.fixed_size.json \
    -plot -plotfname eval.png -verbose
```

Prints a comparison table (one row per extension) with `n_total`, `n_predicted`,
`coverage`, `mean_gt_ppi`, `mae`, `mape`, `rmse`, `median_ae`, `log2_mae`, `log2_rmse`.
With `-plot` a scatter plot of predicted vs GT PPI is displayed or saved.
