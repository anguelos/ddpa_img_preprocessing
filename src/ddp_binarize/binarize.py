from abc import ABC, abstractmethod
from PIL import Image
import numpy as np


class Binarizer(ABC):
    @abstractmethod
    def __call__(self, img):
        pass


class OtsuBinarizer(Binarizer):
    def __call__(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        gray = np.array(img.convert("L"))
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        hist = hist.astype(float)
        total = gray.size
        sum_total = float(np.dot(np.arange(256), hist))
        w_bg, sum_bg, max_var, threshold = 0.0, 0.0, 0.0, 0
        for t in range(256):
            w_bg += hist[t]
            if w_bg == 0:
                continue
            w_fg = total - w_bg
            if w_fg == 0:
                break
            sum_bg += t * hist[t]
            m_bg = sum_bg / w_bg
            m_fg = (sum_total - sum_bg) / w_fg
            var = w_bg * w_fg * (m_bg - m_fg) ** 2
            if var > max_var:
                max_var = var
                threshold = t
        result = np.where(gray <= threshold, 0, 255).astype(np.uint8)
        return Image.fromarray(result, mode="L")


def main_binarize_offline():
    import fargv
    import sys
    import tqdm

    p = {
        "images": set([]),
        "method": [("otsu", "bunet"), "Binarization method to use one of [otsu, bunet]"],
        "verbose": False,
    }
    args, _ = fargv.fargv(p)
    if args.method == "otsu":
        binarizer = OtsuBinarizer()
    elif args.method == "bunet":
        raise NotImplementedError("bunet binarizer is not yet implemented")

    for img_path in tqdm.tqdm(args.images, disable=not args.verbose):
        img = Image.open(img_path)
        if args.verbose:
            print(f"Binarizing {img_path} ({img.size}) using {args.method} method...")
        bin_img = binarizer(img)
        # img_md5_sum.img.ext -> img_md5_sum.bin.png
        parts = img_path.split(".img.")
        out_path = parts[0] + ".bin.png"
        bin_img.save(out_path)
        if args.verbose:
            print(f"{img_path} -> {out_path}", file=sys.stderr)
