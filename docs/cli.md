# Command Line Tools

## ddp_binarize_offline

Binarizes charter images. Output is grayscale where 0=foreground (text), 255=background.

```
ddp_binarize_offline <image> [<image> ...] [-method otsu|bunet] [-verbose]
```

Output file: `<img_md5>.bin.png` (same directory as input)

## ddp_resolution_offline

Estimates pixels-per-inch of charter images.

```
ddp_resolution_offline <image> [<image> ...] [-verbose] [-save_resized] [-target_ppi 300.0] [-max_size 4096] [-min_size 256]
```

Output file: `<img_md5>.resolution.json`

## ddp_recto_offline

Selects the most appropriate image (recto) for each charter directory.

```
ddp_recto_offline <charter_dir1> [<charter_dir2> ...] [-nosymlinks] [-min_recto_prob 0.4] [-verbose]
```

Creates symlink `CH.recto.<ext>` if best image probability >= min_recto_prob.

## ddp_cv_preprocess_offline

Runs the full preprocessing pipeline on one or more FSDB roots.

```
ddp_cv_preprocess_offline <fsdb_root1> [<fsdb_root2> ...] [-nosymlinks] [-min_recto_prob 0.4] [-verbose]
```
