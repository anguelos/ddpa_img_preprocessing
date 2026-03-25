import pytest
import json
from pathlib import Path
from ddp_cv_preprocess.fsdb import FSDBCharter, iter_fsdb, CHARTER_REQUIRED_FILES
from ddp_cv_preprocess.util import FSDBIntegrityException


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


def test_fsdb_charter_atom_id(charter_dir):
    charter = FSDBCharter(charter_dir)
    atom_id = charter.atom_id
    assert isinstance(atom_id, str)
    assert len(atom_id) > 0


def test_fsdb_charter_source_url(charter_dir):
    charter = FSDBCharter(charter_dir)
    url = charter.source_url
    assert url.startswith("http")
