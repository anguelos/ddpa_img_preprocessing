import pytest
import json
import os
from ddp_recto.recto_verso import rank_images, HeuristicRectoSelector, RectoSelector
from ddp_cv_preprocess.util import FSDBIntegrityException
import inspect


def test_rank_images_returns_list(charter_dir):
    result = rank_images(str(charter_dir))
    assert isinstance(result, list)


def test_rank_images_nonempty(charter_dir):
    result = rank_images(str(charter_dir))
    assert len(result) > 0


def test_rank_images_structure(charter_dir):
    result = rank_images(str(charter_dir))
    for item in result:
        assert len(item) == 2
        path, prob = item
        assert isinstance(path, str)
        assert isinstance(prob, float)


def test_rank_images_sorted_by_prob(charter_dir):
    result = rank_images(str(charter_dir))
    probs = [p for _, p in result]
    assert probs == sorted(probs, reverse=True)


def test_heuristic_recto_selector(charter_dir):
    selector = HeuristicRectoSelector()
    result = selector(str(charter_dir))
    assert isinstance(result, list)
    assert len(result) > 0


def test_recto_selector_is_abstract():
    assert inspect.isabstract(RectoSelector)


def test_rank_images_missing_image(tmp_path):
    charter = tmp_path / "charter"
    charter.mkdir()
    (charter / "CH.cei.xml").write_text("<charter/>")
    (charter / "CH.url.txt").write_text("http://example.com")
    (charter / "CH.atom_id.txt").write_text("test")
    (charter / "image_urls.json").write_text(json.dumps({"nonexistent.img.jpg": "http://example.com/img.jpg"}))
    with pytest.raises(FSDBIntegrityException):
        rank_images(str(charter))
