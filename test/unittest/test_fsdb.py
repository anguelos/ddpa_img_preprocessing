import pytest
import json
from pathlib import Path
from ddp_cv_preprocess.fsdb import FSDBCharter, iter_fsdb, CHARTER_REQUIRED_FILES
from ddp_cv_preprocess.util import FSDBIntegrityException

FAKE_FSDB_ROOT = Path(__file__).parent.parent / "fake_fsdb_root"


def test_fsdb_charter_init(charter_dir):
    charter = FSDBCharter(charter_dir)
    assert charter.charter_dir == Path(charter_dir)


def test_fsdb_charter_image_paths(charter_dir):
    charter = FSDBCharter(charter_dir)
    imgs = charter.image_paths
    assert len(imgs) > 0


def test_fsdb_charter_image_urls(charter_dir):
    charter = FSDBCharter(charter_dir)
    urls = charter.image_urls
    assert isinstance(urls, dict)
    assert len(urls) > 0


def test_fsdb_charter_validate_ok(charter_dir):
    charter = FSDBCharter(charter_dir)
    assert charter.validate()


def test_fsdb_charter_validate_missing_file(tmp_path):
    charter_dir = tmp_path / "charter"
    charter_dir.mkdir()
    charter = FSDBCharter(charter_dir)
    with pytest.raises(FSDBIntegrityException):
        charter.validate()


def test_iter_fsdb(fake_fsdb):
    charters = list(iter_fsdb(fake_fsdb))
    assert len(charters) > 0
    for c in charters:
        assert isinstance(c, FSDBCharter)


def test_iter_fsdb_finds_all_archives(fake_fsdb):
    charters = list(iter_fsdb(fake_fsdb))
    archives = {c.charter_dir.parts[-3] for c in charters}
    assert "Book-Archive" in archives
    assert "Paper-Charter" in archives
    assert "TypicalCharter" in archives
    assert "TESTARCH" in archives


def test_fsdb_charter_atom_id(charter_dir):
    charter = FSDBCharter(charter_dir)
    assert isinstance(charter.atom_id, str) and len(charter.atom_id) > 0


def test_fsdb_charter_source_url(charter_dir):
    charter = FSDBCharter(charter_dir)
    assert charter.source_url.startswith("http")


# --- real-world charter tests ---

def test_book_charter_has_multiple_images(book_charter_dir):
    charter = FSDBCharter(book_charter_dir)
    assert len(charter.image_paths) == 5


def test_book_charter_image_urls_without_img_infix(book_charter_dir):
    # Real data: keys are md5.png not md5.img.png
    charter = FSDBCharter(book_charter_dir)
    for key in charter.image_urls:
        assert ".img." not in key


def test_book_charter_validates(book_charter_dir):
    assert FSDBCharter(book_charter_dir).validate()


def test_paper_charter_validates(paper_charter_dir):
    assert FSDBCharter(paper_charter_dir).validate()


def test_typical_charter_has_two_images(typical_charter_dir):
    charter = FSDBCharter(typical_charter_dir)
    assert len(charter.image_paths) == 2


def test_typical_charter_validates(typical_charter_dir):
    assert FSDBCharter(typical_charter_dir).validate()


def test_charters_with_layout_pred(book_charter_dir):
    charter = FSDBCharter(book_charter_dir)
    for img_path in charter.image_paths:
        pred_path = charter.layout_pred_path(img_path)
        assert pred_path.exists()


def test_layout_pred_json_schema(book_charter_dir):
    charter = FSDBCharter(book_charter_dir)
    img_path = charter.image_paths[0]
    pred = json.loads(charter.layout_pred_path(img_path).read_text())
    assert "class_names" in pred
    assert "rect_classes" in pred
    assert "rect_LTRB" in pred
    assert len(pred["rect_classes"]) == len(pred["rect_LTRB"])
    assert "Img:WritableArea" in pred["class_names"]
    assert "Wr:OldText" in pred["class_names"]
