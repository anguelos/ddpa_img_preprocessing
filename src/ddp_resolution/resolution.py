import json
import os
from abc import ABC, abstractmethod
from PIL import Image


class ResolutionEstimator(ABC):
    @abstractmethod
    def __call__(self, img):
        pass


class HeuristicResolutionEstimator(ResolutionEstimator):
    DEFAULT_PPI = 300.0
    TYPICAL_CHARTER_WIDTH_CM = 30.0

    def __call__(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        if "dpi" in img.info:
            dpi = img.info["dpi"]
            ppi = (dpi[0] + dpi[1]) / 2.0
            if ppi > 0:
                return float(ppi), 0.9
        w, _ = img.size
        estimated_ppi = w / (self.TYPICAL_CHARTER_WIDTH_CM / 2.54)
        return float(estimated_ppi), 0.2


def main_resolution_offline():
    import fargv
    import tqdm

    p = {
        "images": set([]),
        "save_resized": False,
        "target_ppi": 300.0,
        "max_size": 4096,
        "min_size": 256,
        "verbose": False,
    }
    args, _ = fargv.fargv(p)
    estimator = HeuristicResolutionEstimator()
    for img_path in tqdm.tqdm(args.images, disable=not args.verbose):
        img = Image.open(img_path)
        ppi, confidence = estimator(img)
        parts = img_path.split(".img.")
        json_path = parts[0] + ".resolution.json"
        with open(json_path, "w") as f:
            json.dump({"ppi": ppi, "confidence": confidence}, f)
        if args.save_resized:
            scale = args.target_ppi / ppi
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            new_w = max(args.min_size, min(args.max_size, new_w))
            new_h = max(args.min_size, min(args.max_size, new_h))
            resized = img.resize((new_w, new_h))
            ext = img_path.split(".img.")[-1]
            n = int(round(args.target_ppi))
            out_path = parts[0] + f".scaled_ppi_{n}.png"
            resized.save(out_path)
