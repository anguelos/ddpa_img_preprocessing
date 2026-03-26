# DDPA Image Preprocessing — Design Reference

## Project Purpose
Preprocessing pipeline for medieval charter images stored in FSDB (Filesystem Database).
Three main tasks: binarization, resolution estimation, recto/verso selection.

## FSDB Architecture
Hierarchy: `root/Archive/Fond/Charter`
- Archive: alphanumeric name (e.g. `AT-ADG`)
- Fond: md5 hash directory name
- Charter: md5 hash directory name
- Charter contents:
  - `CH.cei.xml` — transcription/metadata XML
  - `CH.url.txt` — source URL
  - `CH.atom_id.txt` — unique atom ID used to derive charter's md5 name
  - `image_urls.json` — maps image filename to source URL
  - `<img_md5>.img.<ext>` — immutable image files
  - `<img_md5>.layout.pred.json` — optional YOLO layout prediction
  - `<img_md5>.layout.gt.json` — optional manual layout annotation
- Note: image_urls.json may reference images as `<md5>.<ext>` instead of `<md5>.img.<ext>` (inconsistency to handle)

## Module Structure
```
src/
  ddp_binarize/     — image binarization (output: 0=foreground, 255=background)
  ddp_resolution/   — PPI estimation
  ddp_recto/        — select best charter image (recto)
  ddp_cv_preprocess/ — FSDB utilities + offline pipeline entry point
```

## Functor Interface Convention
Each task has:
1. Abstract base class (ABC) with `__call__` method
2. `Heuristic*` concrete implementation using classical CV
3. Future ML implementation inheriting from same ABC, using `torch_mentor.Mentee`

Example:
```python
class Binarizer(ABC):
    @abstractmethod
    def __call__(self, img): ...

class OtsuBinarizer(Binarizer):
    def __call__(self, img): ...
```

## Entry Points
- `ddp_binarize_offline` → `ddp_binarize.binarize:main_binarize_offline`
- `ddp_resolution_offline` → `ddp_resolution.resolution:main_resolution_offline`
- `ddp_recto_offline` → `ddp_recto.recto_verso:main_recto_verso_offline`
- `ddp_cv_preprocess_offline` → `ddp_cv_preprocess.offline:main_cv_preprocess_offline`

## Argument Parsing (fargv)
All parameters use a **single dash** (e.g. `-verbose`, `-method`, `-nosymlinks`).

| fargv value | CLI type | Example |
|---|---|---|
| `set([default])` | named, multi-value | `p = {"images": set([])}` → `script.py -images img1.jpg img2.jpg` |
| `("default", "opt2", ...)` | choice (first = default) | `p = {"method": ("otsu", "bunet")}` → `-method bunet` |
| `False` | boolean flag, default off | `p = {"verbose": False}` → `-verbose` to enable |
| `True` | boolean flag, default on | `p = {"verbose": True}` → `-noverbose` to disable |
| `scalar` | named arg with default | `p = {"min_prob": 0.4}` → `-min_prob 0.2` |

Example from `ddp_binarize_offline`:
```python
p = {
    "images": set([]),
    "method": ("otsu", "bunet"),
    "verbose": False,
}
args, _ = fargv.fargv(p)
# invoked as: ddp_binarize_offline img.jpg -method bunet -verbose
```

## Output Convention
- All user-facing messages (verbose output, warnings, errors) go to `sys.stderr`
- `stdout` is reserved for data output only (e.g. piped results)
- Always pass `file=sys.stderr` to `print()`

## torch_mentor Integration
- ML functors inherit from `mentor.Mentee` (from `../torch_mentor`)
- Heuristic functors are standalone (no torch dependency)
- Training utilities live in `main_train_*` entry points
- See `src/ddp_binarize/binet.py` and `src/ddp_resolution/resresnet.py` for ML scaffolds

## Binarization Output Convention
- Pixel value 0 = certainly foreground (text)
- Pixel value 255 = certainly background
- Values < 128 = probably foreground
- Values > 128 = probably background
- Output is always grayscale PNG, named `<img_md5>.bin.png`

## Resolution Output
- Returns (estimated_ppi, confidence) tuple
- confidence in [0, 1]
- Output JSON named `<img_md5>.resolution.json`

## Recto Selection
- Returns list of (image_path, probability) sorted by probability descending
- probability < 0.5 for all images means no good recto found
- Creates symlink `CH.recto.<ext>` pointing to best image

## Decisions Log
- 2026-03-25: Project initialized from CLAUDE_INITIAL.md scaffold
- Import fix: ddp_recto.recto_verso uses `from ddp_cv_preprocess.util import FSDBIntegrityException`
- Package install layout: `src/` layout with `package_dir={'':' src'}`
- License: AGPL-3.0 (Affero Public License)
## Layout File Schema (`.layout.pred.json` / `.layout.gt.json`)
Produced by a YOLO object detector, one file per image.

```json
{
  "img_md5": "<md5>",
  "class_names": ["No Class", "Ignore", "Img:CalibrationCard", "Img:Seal",
                  "Img:WritableArea", "Wr:OldText", "Wr:OldNote", "Wr:NewText",
                  "Wr:NewOther", "WrO:Ornament", "WrO:Fold"],
  "image_wh": [width, height],
  "rect_LTRB": [[left, top, right, bottom], ...],
  "rect_captions": ["$conf:0.82", ...],
  "rect_classes": [4, 5, ...]
}
```

Key class indices: `2`=CalibrationCard, `3`=Seal, `4`=WritableArea, `5`=OldText, `9`=Ornament

## Real-world FSDB Notes
- `image_urls.json` keys are typically `md5.ext` (without `.img.`) — the `.img.` infix variant is less common
- Book scans may have 5+ images (consecutive pages), no dominant recto
- Typical charters have 2 images: recto + verso
- All images in `test/fake_fsdb_root/` except TESTARCH have layout predictions available
