import json
import os
import sys
import tqdm

from ddp_cv_preprocess.fsdb import FSDBCharter, iter_fsdb
from ddp_cv_preprocess.util import FSDBIntegrityException
from ddp_binarize.binarize import OtsuBinarizer
from ddp_resolution.resolution import HeuristicResolutionEstimator
from ddp_recto.recto_verso import HeuristicRectoSelector


def process_charter(charter, binarizer, resolution_estimator, recto_selector, args):
    try:
        charter.validate()
    except FSDBIntegrityException as e:
        print(f"Skipping {charter}: {e}", file=sys.stderr)
        return

    ranked = recto_selector(str(charter.charter_dir))
    if args.verbose:
        print(f"Ranked images for {charter.charter_dir}: {ranked}", file=sys.stderr)

    for img_path in charter.image_paths:
        img_str = str(img_path)
        stem = img_path.name.split(".img.")[0]

        ppi, confidence = resolution_estimator(img_str)
        json_path = charter.charter_dir / f"{stem}.resolution.json"
        with open(json_path, "w") as f:
            json.dump({"ppi": ppi, "confidence": confidence}, f)

        bin_img = binarizer(img_str)
        bin_path = charter.charter_dir / f"{stem}.bin.png"
        bin_img.save(str(bin_path))

    if ranked and ranked[0][1] >= args.min_recto_prob and not args.nosymlinks:
        best_path, _ = ranked[0]
        ext = best_path.split(".img.")[-1] if ".img." in best_path else best_path.split(".")[-1]
        symlink_path = charter.charter_dir / f"CH.recto.{ext}"
        if not symlink_path.exists():
            os.symlink(best_path, str(symlink_path))


def main_cv_preprocess_offline():
    import fargv

    p = {
        "fsdb_root": set([]),
        "min_recto_prob": 0.4,
        "nosymlinks": False,
        "verbose": False,
    }
    args, _ = fargv.fargv(p)
    binarizer = OtsuBinarizer()
    resolution_estimator = HeuristicResolutionEstimator()
    recto_selector = HeuristicRectoSelector()

    for fsdb_root in args.fsdb_root:
        charters = list(iter_fsdb(fsdb_root))
        for charter in tqdm.tqdm(charters, disable=not args.verbose):
            process_charter(charter, binarizer, resolution_estimator, recto_selector, args)
