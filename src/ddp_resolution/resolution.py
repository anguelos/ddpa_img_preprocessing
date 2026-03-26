"""Resolution estimation from image files and layout JSON annotations."""

import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from PIL import Image


CALIB_CARD_LONG_CM: float = 19.5
"""Long dimension of the calibration card in centimetres."""

CALIB_CARD_SHORT_CM: float = 6.0
"""Short dimension of the calibration card in centimetres (midpoint of 5–7 cm range)."""


class ResolutionEstimator(ABC):
    """Abstract base class for image resolution estimators.

    All concrete estimators must implement :meth:`__call__` and return a
    ``(ppi, confidence)`` pair, or *None* when estimation is not possible.
    """

    @abstractmethod
    def __call__(
        self, img: Union[str, Image.Image]
    ) -> Optional[Tuple[float, float]]:
        """Estimate the scan resolution of *img*.

        Parameters
        ----------
        img : str or PIL.Image.Image
            Either a file-system path to the image or an already-opened
            PIL image object.

        Returns
        -------
        tuple of (float, float) or None
            ``(ppi, confidence)`` where *ppi* is pixels-per-inch and
            *confidence* is in ``[0, 1]``.  Returns *None* when
            estimation is not possible.
        """


class FixedSizeResolutionEstimator(ResolutionEstimator):
    """Estimate resolution by assuming a fixed physical charter size.

    The largest image dimension is divided by :attr:`CHARTER_LARGEST_DIM_CM`
    to derive PPI.  This is a low-confidence fallback when no calibration
    artefact is available.

    Attributes
    ----------
    CHARTER_LARGEST_DIM_CM : float
        Assumed physical length of the longest charter edge in centimetres.
    """

    CHARTER_LARGEST_DIM_CM: float = 42.0

    def __call__(
        self, img: Union[str, Image.Image]
    ) -> Tuple[float, float]:
        """Estimate PPI from the largest image dimension.

        Parameters
        ----------
        img : str or PIL.Image.Image
            Image path or open PIL image.

        Returns
        -------
        tuple of (float, float)
            ``(ppi, 0.2)`` — fixed confidence of 0.2 reflects the low
            reliability of the assumption.

        Examples
        --------
        >>> from PIL import Image
        >>> est = FixedSizeResolutionEstimator()
        >>> img = Image.new("RGB", (3508, 4961))  # A3 at 300 dpi
        >>> ppi, conf = est(img)
        >>> round(ppi)
        300
        >>> conf
        0.2
        """
        if isinstance(img, str):
            img = Image.open(img)
        ppi = max(img.size) / (self.CHARTER_LARGEST_DIM_CM / 2.54)
        return float(ppi), 0.2


class CalibrationCardResolutionEstimator(ResolutionEstimator):
    """Estimate resolution from a calibration card detected in a layout file.

    The estimator looks for an ``Img:CalibrationCard`` region in a layout
    JSON file co-located with the image.  Both ground-truth and predicted
    layout files are supported.

    Parameters
    ----------
    use_gt_layout : bool, optional
        When *True*, reads ``.layout.gt.json``; when *False* (default),
        reads ``.layout.pred.json``.

    Attributes
    ----------
    _layout_suffix : str
        File suffix appended to the image stem to locate the layout file.
    """

    def __init__(self, use_gt_layout: bool = False) -> None:
        self._layout_suffix: str = (
            ".layout.gt.json" if use_gt_layout else ".layout.pred.json"
        )

    def __call__(self, img: str) -> Optional[Tuple[float, float]]:
        """Estimate PPI from the calibration card in the associated layout file.

        Parameters
        ----------
        img : str
            File-system path to the image.  The layout file is derived by
            replacing the ``.img.<ext>`` suffix with
            ``self._layout_suffix``.

        Returns
        -------
        tuple of (float, float) or None
            ``(ppi, 0.85)`` on success.  Returns *None* when the layout
            file is absent or contains no ``Img:CalibrationCard`` region.

        Raises
        ------
        ValueError
            If *img* is not a string path.

        Examples
        --------
        >>> est = CalibrationCardResolutionEstimator()
        >>> est("/data/charter/abc.img.jpg")  # doctest: +SKIP
        (312.4, 0.85)
        """
        if not isinstance(img, str):
            raise ValueError(
                "CalibrationCardResolutionEstimator requires an image path, not a PIL image"
            )
        img_path: str = img
        layout_path: str = img_path.split(".img.")[0] + self._layout_suffix
        if not os.path.exists(layout_path):
            return None
        return self._estimate_from_layout(layout_path)

    def _estimate_from_layout(
        self, layout_path: str
    ) -> Optional[Tuple[float, float]]:
        """Compute PPI from calibration-card rectangles in a layout JSON file.

        Parameters
        ----------
        layout_path : str
            Path to the layout JSON file containing ``class_names``,
            ``rect_LTRB``, and ``rect_classes`` keys.

        Returns
        -------
        tuple of (float, float) or None
            ``(mean_ppi, 0.85)`` averaged over all detected calibration
            cards.  Returns *None* when no calibration card is found.
        """
        layout: dict = json.load(open(layout_path))
        calib_idx: int = layout["class_names"].index("Img:CalibrationCard")
        calib_rects = [
            rect
            for rect, cls in zip(layout["rect_LTRB"], layout["rect_classes"])
            if cls == calib_idx
        ]
        if not calib_rects:
            return None
        ppis: list[float] = []
        for L, T, R, B in calib_rects:
            long_px: float = max(R - L, B - T)
            short_px: float = min(R - L, B - T)
            ppi_long: float = long_px / (CALIB_CARD_LONG_CM / 2.54)
            ppi_short: float = short_px / (CALIB_CARD_SHORT_CM / 2.54)
            ppis.append((ppi_long + ppi_short) / 2.0)
        return float(sum(ppis) / len(ppis)), 0.85


def main_resolution_offline() -> None:
    """CLI entry-point for offline per-image resolution estimation.

    Reads image paths from ``images``, runs the selected estimator, and
    writes a JSON prediction file next to each image.  Exits with code 1
    if any image could not be processed.

    The output file name is derived automatically from the method unless
    ``out_ext`` is set explicitly:

    ============  =========================
    method        default out_ext
    ============  =========================
    layout        .res.pred_layout.json
    layoutgt      .res.pred_layoutgt.json
    fixed_size    .res.pred_fixed.json
    resresnet     .res.pred_resresnet.json
    ============  =========================
    """
    import fargv
    import sys
    import tqdm

    p = {
        "images": set([]),
        "method": [
            ("layout", "layoutgt", "fixed_size", "resresnet"),
            "Resolution estimation method to use one of [layout, layoutgt, fixed_size, resresnet]",
        ],
        "save_resized": False,
        "target_ppi": 300.0,
        "max_size": 4096,
        "min_size": 256,
        "out_ext": "",
        "verbose": False,
    }
    args, _ = fargv.fargv(p)

    _DEFAULT_EXT: dict[str, str] = {
        "layout": ".res.pred_layout.json",
        "layoutgt": ".res.pred_layoutgt.json",
        "fixed_size": ".res.pred_fixed.json",
        "resresnet": ".res.pred_resresnet.json",
    }

    estimator: ResolutionEstimator
    if args.method == "layout":
        estimator = CalibrationCardResolutionEstimator(use_gt_layout=False)
    elif args.method == "layoutgt":
        estimator = CalibrationCardResolutionEstimator(use_gt_layout=True)
    elif args.method == "fixed_size":
        estimator = FixedSizeResolutionEstimator()
    elif args.method == "resresnet":
        raise NotImplementedError("resresnet resolution estimator is not yet implemented")

    out_ext: str = args.out_ext if args.out_ext else _DEFAULT_EXT[args.method]

    failed_images: list[tuple[str, str]] = []
    successful_ppis: list[float] = []
    for img_path in tqdm.tqdm(args.images, disable=not args.verbose):
        try:
            result: Optional[Tuple[float, float]] = estimator(
                img_path if args.method in ("layout", "layoutgt") else Image.open(img_path)
            )
            if result is None:
                failed_images.append((img_path, "no layout file or no Img:CalibrationCard"))
                continue
            ppi, confidence = result
            successful_ppis.append(ppi)
            parts = img_path.split(".img.")
            json_path: str = parts[0] + out_ext
            with open(json_path, "w") as f:
                json.dump({"ppi": ppi, "confidence": confidence}, f)
            if args.verbose:
                tqdm.tqdm.write(
                    f"{os.path.basename(img_path)} -> ppi={ppi:.1f} confidence={confidence:.2f}",
                    file=sys.stderr,
                )
            if args.save_resized:
                img = Image.open(img_path)
                img.load()
                scale: float = args.target_ppi / ppi
                new_w: int = max(args.min_size, min(args.max_size, int(img.width * scale)))
                new_h: int = max(args.min_size, min(args.max_size, int(img.height * scale)))
                n: int = int(round(args.target_ppi))
                out_path: str = parts[0] + f".scaled_ppi_{n}.png"
                img.resize((new_w, new_h)).save(out_path)
        except Exception as e:
            failed_images.append((img_path, str(e)))

    if args.verbose:
        if failed_images:
            print(f"\nFailed images ({len(failed_images)}):", file=sys.stderr)
            for i, (path, reason) in enumerate(failed_images, 1):
                print(f"  {i}. {path}: {reason}", file=sys.stderr)
        n_ok: int = len(successful_ppis)
        avg_ppi: float = sum(successful_ppis) / n_ok if n_ok else float("nan")
        print(f"\nSuccessful: {n_ok}  avg PPI: {avg_ppi:.1f}", file=sys.stderr)
    if failed_images:
        sys.exit(1)
