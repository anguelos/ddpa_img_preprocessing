import pytest
import numpy as np
from PIL import Image
from ddp_binarize.binarize import OtsuBinarizer, Binarizer


def test_otsu_binarizer_returns_grayscale(sample_image):
    binarizer = OtsuBinarizer()
    result = binarizer(sample_image)
    assert result.mode == 'L'


def test_otsu_binarizer_same_size(sample_image):
    binarizer = OtsuBinarizer()
    result = binarizer(sample_image)
    assert result.size == sample_image.size


def test_otsu_binarizer_from_path(sample_image_path):
    binarizer = OtsuBinarizer()
    result = binarizer(sample_image_path)
    assert result.mode == 'L'


def test_otsu_binarizer_output_range(sample_image):
    binarizer = OtsuBinarizer()
    result = binarizer(sample_image)
    arr = np.array(result)
    assert arr.min() >= 0
    assert arr.max() <= 255


def test_otsu_binarizer_dark_image_is_foreground():
    dark = Image.new('L', (10, 10), color=10)
    light = Image.new('L', (10, 10), color=245)
    mixed = Image.fromarray(
        np.concatenate([np.full((10, 5), 10, dtype=np.uint8),
                        np.full((10, 5), 245, dtype=np.uint8)], axis=1), mode='L'
    )
    binarizer = OtsuBinarizer()
    result = binarizer(mixed)
    arr = np.array(result)
    assert arr[:, :5].mean() < 128
    assert arr[:, 5:].mean() > 128


def test_binarizer_is_abstract():
    import inspect
    assert inspect.isabstract(Binarizer)
