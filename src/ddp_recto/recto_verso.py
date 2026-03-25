import tqdm
import json
import os
import sys
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import List, Tuple

from ddp_cv_preprocess.util import FSDBIntegrityException


class RectoSelector(ABC):
    @abstractmethod
    def __call__(self, charter_dir):
        pass


def rank_images(charter_dir):
    image_paths = [Path(f).name for f in glob(f"{charter_dir}/*.img.*")]
    image_id_to_name = {p.split(".img.")[0]: p for p in image_paths}
    raw_urls = json.load(open(f"{charter_dir}/image_urls.json"))
    # Normalise keys: handle both img_md5.img.ext and img_md5.ext
    image_urls = {}
    for k, v in raw_urls.items():
        img_id = k.split(".img.")[0].split(".")[0]
        image_urls[img_id] = v
    image_id_scores = []
    keys = list(image_urls.keys())
    for n, img_id in enumerate(reversed(keys)):
        if img_id not in image_id_to_name:
            raise FSDBIntegrityException(f"Missing image {img_id} in {charter_dir}")
        prob = n / (1 + len(image_urls))
        image_id_scores.append((prob, img_id))
    return [[image_id_to_name[img_id], prob] for prob, img_id in sorted(image_id_scores, reverse=True)]


class HeuristicRectoSelector(RectoSelector):
    def __call__(self, charter_dir):
        return rank_images(charter_dir)


def main_recto_verso_offline():
    import fargv

    p = {
        "charter_dir": set([]),
        "nosymlinks": False,
        "min_recto_prob": 0.4,
        "verbose": False,
    }
    args, _ = fargv.fargv(p)
    selector = HeuristicRectoSelector()
    for charter in tqdm.tqdm(args.charter_dir, disable=not args.verbose):
        try:
            ranked = selector(charter)
        except FSDBIntegrityException as e:
            print(f"Integrity error: {e}", file=sys.stderr)
            continue
        if not ranked:
            print(f"No images found in {charter}", file=sys.stderr)
            continue
        best_path, best_prob = ranked[0]
        if best_prob < args.min_recto_prob:
            print(f"Warning: low confidence ({best_prob:.2f}) for {charter}", file=sys.stderr)
        if not args.nosymlinks and best_prob >= args.min_recto_prob:
            ext = best_path.split(".img.")[-1] if ".img." in best_path else best_path.split(".")[-1]
            symlink_path = os.path.join(charter, f"CH.recto.{ext}")
            if not os.path.exists(symlink_path):
                os.symlink(best_path, symlink_path)
