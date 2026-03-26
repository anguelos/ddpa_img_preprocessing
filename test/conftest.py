import json
import pytest
from pathlib import Path
from PIL import Image

FAKE_FSDB_ROOT = Path(__file__).parent / "fake_fsdb_root"

# --- synthetic TESTARCH charter (img key uses .img. variant) ---
ARCHIVE = "TESTARCH"
FOND_MD5 = "a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5"
CHARTER_MD5 = "b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6"
IMAGE_MD5 = "c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7"
IMAGE_FILENAME = f"{IMAGE_MD5}.img.jpg"

# --- real-world charter paths ---
BOOK_CHARTER_DIR = FAKE_FSDB_ROOT / "Book-Archive" / "77d1af4724ed5bf69180105c278b1342" / "aaa8343a31960496af8d6b59337ffafe"
PAPER_CHARTER_DIR = FAKE_FSDB_ROOT / "Paper-Charter" / "eb8d75f7c2a1f46d3c919ef8267e88a3" / "aaa1f5632a8eff5a7143ae73d78dc22a"
TYPICAL_CHARTER_DIR = FAKE_FSDB_ROOT / "TypicalCharter" / "2de82af2d01c041e6bb08d53d322f016" / "aaa8dfe9b6426e95026e3b9f7f2d2d95"


@pytest.fixture(scope="session", autouse=True)
def fake_fsdb():
    # Ensure the synthetic TESTARCH image exists (created if missing)
    charter_dir = FAKE_FSDB_ROOT / ARCHIVE / FOND_MD5 / CHARTER_MD5
    charter_dir.mkdir(parents=True, exist_ok=True)
    img_path = charter_dir / IMAGE_FILENAME
    if not img_path.exists():
        Image.new("RGB", (200, 300), color=(220, 210, 190)).save(str(img_path))
    return FAKE_FSDB_ROOT


@pytest.fixture(scope="session")
def charter_dir(fake_fsdb):
    return fake_fsdb / ARCHIVE / FOND_MD5 / CHARTER_MD5


@pytest.fixture(scope="session")
def sample_image_path(charter_dir):
    return str(charter_dir / IMAGE_FILENAME)


@pytest.fixture(scope="session")
def sample_image(sample_image_path):
    return Image.open(sample_image_path)


@pytest.fixture(scope="session")
def book_charter_dir():
    return BOOK_CHARTER_DIR


@pytest.fixture(scope="session")
def paper_charter_dir():
    return PAPER_CHARTER_DIR


@pytest.fixture(scope="session")
def typical_charter_dir():
    return TYPICAL_CHARTER_DIR


# --- ResDs fixture ---

@pytest.fixture(scope="session")
def res_ds_root(tmp_path_factory):
    import json
    root = tmp_path_factory.mktemp("res_ds")
    img_md5 = "d" * 32
    charter_dir = root / "RESARCH" / ("a" * 32) / ("b" * 32)
    charter_dir.mkdir(parents=True)

    img = Image.new("RGB", (300, 400), color=(200, 190, 180))
    img.save(str(charter_dir / f"{img_md5}.img.jpg"))

    (charter_dir / f"{img_md5}.layout.pred.json").write_text(json.dumps({
        "class_names": ["No Class", "Ignore", "Img:CalibrationCard", "Img:Seal",
                        "Img:WritableArea", "Wr:OldText", "Wr:OldNote", "Wr:NewText",
                        "Wr:NewOther", "WrO:Ornament", "WrO:Fold"],
        "image_wh": [300, 400],
        "rect_LTRB": [[10, 20, 290, 380], [20, 30, 280, 370]],
        "rect_classes": [4, 5],
    }))

    crops_dir = charter_dir / f"{img_md5}.layout.crops"
    crops_dir.mkdir()
    crop_img = Image.new("RGB", (280, 360), color=(200, 190, 180))
    crop_img.save(str(crops_dir / "0.Img_WritableArea.jpg"))
    (crops_dir / "0.resolution.gt.json").write_text(json.dumps({
        "image_wh": [280, 360],
        "rect_LTRB": [[130, 10, 134, 170]],
        "rect_classes": [0],
        "class_names": ["5cm", "1cm", "5in", "1in", "Undefined"],
        "rect_captions": [""],
        "img_md5": "e" * 32,
    }))
    return root
