import json
import os
import pytest
from pathlib import Path
from PIL import Image

FAKE_FSDB_ROOT = Path(__file__).parent / "fake_fsdb_root"
ARCHIVE = "TESTARCH"
FOND_MD5 = "a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5"
CHARTER_MD5 = "b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6"
IMAGE_MD5 = "c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7"
IMAGE_EXT = "jpg"
IMAGE_FILENAME = f"{IMAGE_MD5}.img.{IMAGE_EXT}"


@pytest.fixture(scope="session", autouse=True)
def fake_fsdb(tmp_path_factory):
    root = tmp_path_factory.mktemp("fsdb") / "fake_fsdb_root"
    charter_dir = root / ARCHIVE / FOND_MD5 / CHARTER_MD5
    charter_dir.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (200, 300), color=(220, 210, 190))
    img.save(str(charter_dir / IMAGE_FILENAME))

    (charter_dir / "CH.cei.xml").write_text("<charter><date>1250</date></charter>")
    (charter_dir / "CH.url.txt").write_text("http://example.com/charter/test")
    (charter_dir / "CH.atom_id.txt").write_text("test_charter_atom_id_12345")
    (charter_dir / "image_urls.json").write_text(json.dumps({
        IMAGE_FILENAME: "http://example.com/images/test.jpg"
    }))

    return root


@pytest.fixture(scope="session")
def charter_dir(fake_fsdb):
    return fake_fsdb / ARCHIVE / FOND_MD5 / CHARTER_MD5


@pytest.fixture(scope="session")
def sample_image_path(charter_dir):
    return str(charter_dir / IMAGE_FILENAME)


@pytest.fixture(scope="session")
def sample_image(sample_image_path):
    return Image.open(sample_image_path)
