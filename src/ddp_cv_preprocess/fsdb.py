import json
import os
from glob import glob
from pathlib import Path
from typing import List, Tuple, Iterator

from ddp_cv_preprocess.util import FSDBIntegrityException

CHARTER_REQUIRED_FILES = ["CH.cei.xml", "CH.url.txt", "CH.atom_id.txt", "image_urls.json"]


class FSDBCharter:
    def __init__(self, charter_dir):
        self.charter_dir = Path(charter_dir)

    @property
    def image_paths(self):
        return sorted(self.charter_dir.glob("*.img.*"))

    @property
    def image_urls(self):
        p = self.charter_dir / "image_urls.json"
        return json.load(open(p))

    @property
    def xml_path(self):
        return self.charter_dir / "CH.cei.xml"

    @property
    def atom_id(self):
        return (self.charter_dir / "CH.atom_id.txt").read_text().strip()

    @property
    def source_url(self):
        return (self.charter_dir / "CH.url.txt").read_text().strip()

    def validate(self):
        for fname in CHARTER_REQUIRED_FILES:
            if not (self.charter_dir / fname).exists():
                raise FSDBIntegrityException(f"Missing {fname} in {self.charter_dir}")
        url_keys = set(self.image_urls.keys())
        img_stems = {p.name.split(".img.")[0] for p in self.image_paths}
        # Normalise url keys to stem only
        url_stems = {k.split(".img.")[0].split(".")[0] for k in url_keys}
        missing = url_stems - img_stems
        if missing:
            raise FSDBIntegrityException(f"image_urls.json references missing images {missing} in {self.charter_dir}")
        return True

    def layout_pred_path(self, img_path):
        stem = Path(img_path).name.split(".img.")[0]
        return self.charter_dir / f"{stem}.layout.pred.json"

    def layout_gt_path(self, img_path):
        stem = Path(img_path).name.split(".img.")[0]
        return self.charter_dir / f"{stem}.layout.gt.json"

    def __repr__(self):
        return f"FSDBCharter({self.charter_dir})"


def iter_fsdb(fsdb_root, validate=False):
    root = Path(fsdb_root)
    for archive in sorted(root.iterdir()):
        if not archive.is_dir():
            continue
        for fond in sorted(archive.iterdir()):
            if not fond.is_dir():
                continue
            for charter_dir in sorted(fond.iterdir()):
                if not charter_dir.is_dir():
                    continue
                charter = FSDBCharter(charter_dir)
                if validate:
                    try:
                        charter.validate()
                    except FSDBIntegrityException:
                        continue
                yield charter
