import pytest
import json
from PIL import Image
from ddp_resolution.resolution import HeuristicResolutionEstimator, ResolutionEstimator
import inspect


def test_heuristic_estimator_returns_tuple(sample_image):
    estimator = HeuristicResolutionEstimator()
    result = estimator(sample_image)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_heuristic_estimator_positive_ppi(sample_image):
    estimator = HeuristicResolutionEstimator()
    ppi, confidence = estimator(sample_image)
    assert ppi > 0


def test_heuristic_estimator_confidence_range(sample_image):
    estimator = HeuristicResolutionEstimator()
    ppi, confidence = estimator(sample_image)
    assert 0.0 <= confidence <= 1.0


def test_heuristic_estimator_from_path(sample_image_path):
    estimator = HeuristicResolutionEstimator()
    ppi, confidence = estimator(sample_image_path)
    assert ppi > 0


def test_heuristic_estimator_uses_dpi_metadata():
    img = Image.new('RGB', (300, 400))
    img.info['dpi'] = (150.0, 150.0)
    estimator = HeuristicResolutionEstimator()
    ppi, confidence = estimator(img)
    assert abs(ppi - 150.0) < 1.0
    assert confidence > 0.5


def test_resolution_estimator_is_abstract():
    assert inspect.isabstract(ResolutionEstimator)
