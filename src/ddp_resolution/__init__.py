"""Resolution estimation package for digitised charter images.

This package provides estimators, a PyTorch dataset, and CLI tools for
estimating and evaluating the scan resolution (PPI) of historical document
images stored in an FSDB archive.

Exported names
--------------
ResolutionEstimator
    Abstract base class for all estimators.
FixedSizeResolutionEstimator
    Low-confidence estimator based on assumed physical charter size.
CalibrationCardResolutionEstimator
    High-confidence estimator based on a detected calibration card.
ResDs
    PyTorch ``Dataset`` pairing charter images with GT PPI labels.
make_train_transform
    Factory for training image transforms.
make_inference_transform
    Factory for inference image transforms.
IMAGENET_MEAN
    Per-channel ImageNet normalisation mean.
IMAGENET_STD
    Per-channel ImageNet normalisation standard deviation.
"""

from ddp_resolution.resolution import (
    ResolutionEstimator,
    FixedSizeResolutionEstimator,
    CalibrationCardResolutionEstimator,
)
from ddp_resolution.res_ds import (
    ResDs,
    make_train_transform,
    make_inference_transform,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

__all__ = [
    "ResolutionEstimator",
    "FixedSizeResolutionEstimator",
    "CalibrationCardResolutionEstimator",
    "ResDs",
    "make_train_transform",
    "make_inference_transform",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
