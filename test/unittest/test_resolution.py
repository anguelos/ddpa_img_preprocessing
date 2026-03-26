import pytest
import json
import os
from pathlib import Path
from PIL import Image
from ddp_resolution.resolution import (
    FixedSizeResolutionEstimator,
    CalibrationCardResolutionEstimator,
    ResolutionEstimator,
    CALIB_CARD_LONG_CM,
    CALIB_CARD_SHORT_CM,
)
import inspect

TYPICAL_RECTO = (
    Path(__file__).parent.parent / "fake_fsdb_root"
    / "TypicalCharter" / "2de82af2d01c041e6bb08d53d322f016"
    / "aaa8dfe9b6426e95026e3b9f7f2d2d95" / "36cfc2d6d7d702c4536d946353c88a14.img.jpg"
)
PAPER_IMG = (
    Path(__file__).parent.parent / "fake_fsdb_root"
    / "Paper-Charter" / "eb8d75f7c2a1f46d3c919ef8267e88a3"
    / "aaa1f5632a8eff5a7143ae73d78dc22a" / "9461c1c22cbfc6f49cc6135592e30a50.img.jpg"
)


def test_resolution_estimator_is_abstract():
    assert inspect.isabstract(ResolutionEstimator)


# --- FixedSizeResolutionEstimator ---

def test_fixed_size_returns_tuple(sample_image):
    est = FixedSizeResolutionEstimator()
    ppi, conf = est(sample_image)
    assert isinstance(ppi, float) and isinstance(conf, float)


def test_fixed_size_positive_ppi(sample_image):
    ppi, _ = FixedSizeResolutionEstimator()(sample_image)
    assert ppi > 0


def test_fixed_size_confidence(sample_image):
    _, conf = FixedSizeResolutionEstimator()(sample_image)
    assert 0.0 <= conf <= 1.0


def test_fixed_size_from_path(sample_image_path):
    ppi, _ = FixedSizeResolutionEstimator()(sample_image_path)
    assert ppi > 0


def test_fixed_size_uses_largest_dimension():
    # 350px wide image at 35cm assumed → 350 / (35/2.54) = 25.4 ppi
    img = Image.new("RGB", (350, 100))
    ppi, _ = FixedSizeResolutionEstimator()(img)
    assert abs(ppi - 25.4) < 0.1


# --- CalibrationCardResolutionEstimator ---

def test_calib_card_returns_tuple(sample_image):
    ppi, conf = CalibrationCardResolutionEstimator()(sample_image)
    assert isinstance(ppi, float) and isinstance(conf, float)


def test_calib_card_falls_back_without_layout(sample_image_path):
    # TESTARCH image has no layout file → falls back to fixed_size (conf=0.2)
    _, conf = CalibrationCardResolutionEstimator()(sample_image_path)
    assert conf == pytest.approx(0.2)


def test_calib_card_uses_layout_when_available():
    # Paper-Charter images have calibration cards detected
    ppi, conf = CalibrationCardResolutionEstimator()(str(PAPER_IMG))
    assert conf == pytest.approx(0.85)
    assert ppi > 0


def test_calib_card_ppi_plausible_for_paper_charter():
    ppi, _ = CalibrationCardResolutionEstimator()(str(PAPER_IMG))
    assert 50 < ppi < 1000


def test_calib_card_typical_recto_has_card():
    ppi, conf = CalibrationCardResolutionEstimator()(str(TYPICAL_RECTO))
    assert conf == pytest.approx(0.85)


def test_calib_card_falls_back_when_no_card_in_layout(book_charter_dir):
    # Book-Archive images have layout files but no calibration cards → falls back
    img_path = str(sorted(book_charter_dir.glob("*.img.*"))[0])
    _, conf = CalibrationCardResolutionEstimator()(img_path)
    assert conf == pytest.approx(0.2)
