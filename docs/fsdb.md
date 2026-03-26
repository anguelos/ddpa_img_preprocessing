# FSDB — Filesystem Database

The FSDB is a hierarchical filesystem structure used to store medieval charter data.

## Hierarchy

```
fsdb_root/
  <Archive>/          — alphanumeric name (e.g. AT-ADG)
    <fond_md5>/       — MD5 hash of the fond identifier
      <charter_md5>/  — MD5 hash of the charter atom ID
```

## Charter Directory Contents

| File | Description |
|---|---|
| `CH.cei.xml` | XML transcription, summary and metadata (CEI format) |
| `CH.url.txt` | Source URL the data was scraped from |
| `CH.atom_id.txt` | Unique atom ID used to derive the charter directory MD5 name |
| `image_urls.json` | Maps image filename to source URL |
| `<img_md5>.img.<ext>` | Immutable image file(s) |
| `<img_md5>.layout.pred.json` | Optional YOLO layout prediction |
| `<img_md5>.layout.gt.json` | Optional manual layout annotation |
| `<img_md5>.bin.png` | Binarized image (produced by this package) |
| `<img_md5>.resolution.json` | PPI estimate (produced by this package) |
| `CH.recto.<ext>` | Symlink to the best recto image (produced by this package) |

## image_urls.json Key Format

Keys follow one of two conventions:

- `<img_md5>.img.<ext>` — includes the `.img.` infix
- `<img_md5>.<ext>` — omits the `.img.` infix (more common in practice)

Both are handled transparently by this package.

## Layout File Schema

Layout files contain bounding box predictions from a YOLO object detector:

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

Key class indices: `2` = CalibrationCard, `3` = Seal, `4` = WritableArea, `5` = OldText.

## Integrity

A valid charter directory must contain all four metadata files (`CH.cei.xml`, `CH.url.txt`,
`CH.atom_id.txt`, `image_urls.json`) and all images referenced in `image_urls.json` must
be present on disk. Use `FSDBCharter.validate()` to check.
