import pytest
from PIL import Image
from ddp_resolution.res_ds import ResDs, _compute_ppi, _dominant_rect


GT_GLOB = "**/*.resolution.gt.json"


# --- unit helpers ---

def test_compute_ppi_5cm():
    gt = {"rect_LTRB": [[130, 10, 134, 170]], "rect_classes": [0],
          "class_names": ["5cm", "1cm", "5in", "1in", "Undefined"]}
    ppi = _compute_ppi(gt, ResDs.KNOWN_SIZES_CM)
    expected = 160 / (5.0 / 2.54)
    assert abs(ppi - expected) < 0.1


def test_compute_ppi_averages_multiple():
    gt = {"rect_LTRB": [[0, 0, 0, 100], [0, 0, 0, 200]], "rect_classes": [1, 1],
          "class_names": ["5cm", "1cm", "5in", "1in", "Undefined"]}
    ppi = _compute_ppi(gt, ResDs.KNOWN_SIZES_CM)
    expected = ((100 / (1.0 / 2.54)) + (200 / (1.0 / 2.54))) / 2
    assert abs(ppi - expected) < 0.1


def test_compute_ppi_skips_undefined():
    gt = {"rect_LTRB": [[0, 0, 0, 100]], "rect_classes": [4],
          "class_names": ["5cm", "1cm", "5in", "1in", "Undefined"]}
    assert _compute_ppi(gt, ResDs.KNOWN_SIZES_CM) is None


def test_dominant_rect_single():
    layout = {"class_names": ["No Class", "Ignore", "Img:CalibrationCard", "Img:Seal", "Img:WritableArea"],
              "rect_LTRB": [[0, 0, 100, 200]], "rect_classes": [4]}
    assert _dominant_rect(layout, "Img:WritableArea", ResDs.DOMINANT_AREA_RATIO) == [0, 0, 100, 200]


def test_dominant_rect_no_class():
    layout = {"class_names": ["No Class", "Ignore", "Img:CalibrationCard", "Img:Seal", "Img:WritableArea"],
              "rect_LTRB": [[0, 0, 100, 200]], "rect_classes": [0]}
    assert _dominant_rect(layout, "Img:WritableArea", ResDs.DOMINANT_AREA_RATIO) is None


def test_dominant_rect_two_equal_not_dominant():
    layout = {"class_names": ["No Class", "Img:WritableArea"],
              "rect_LTRB": [[0, 0, 10, 10], [20, 20, 30, 30]], "rect_classes": [1, 1]}
    assert _dominant_rect(layout, "Img:WritableArea", ResDs.DOMINANT_AREA_RATIO) is None


def test_dominant_rect_one_large_one_small():
    layout = {"class_names": ["No Class", "Img:WritableArea"],
              "rect_LTRB": [[0, 0, 100, 100], [0, 0, 5, 5]], "rect_classes": [1, 1]}
    assert _dominant_rect(layout, "Img:WritableArea", ResDs.DOMINANT_AREA_RATIO) == [0, 0, 100, 100]


# --- ResDs integration ---

def test_res_ds_loads(res_ds_root):
    ds = ResDs.from_root(res_ds_root, GT_GLOB, image_crop="img")
    assert len(ds) == 1


def test_res_ds_getitem_returns_pil_and_ppi(res_ds_root):
    ds = ResDs.from_root(res_ds_root, GT_GLOB, image_crop="img")
    img, ppi = ds[0]
    assert isinstance(img, Image.Image)
    assert isinstance(ppi, float) and ppi > 0


def test_res_ds_writable_area_crop(res_ds_root):
    ds = ResDs.from_root(res_ds_root, GT_GLOB, image_crop="Img:WritableArea")
    assert len(ds) == 1
    img, ppi = ds[0]
    assert img.size == (280, 360)


def test_res_ds_old_text_crop(res_ds_root):
    ds = ResDs.from_root(res_ds_root, GT_GLOB, image_crop="Wr:OldText")
    assert len(ds) == 1
    img, ppi = ds[0]
    assert img.size == (260, 340)


def test_res_ds_with_transform(res_ds_root):
    import torchvision.transforms as T
    import torch
    ds = ResDs.from_root(res_ds_root, GT_GLOB, image_crop="img", input_transform=T.ToTensor())
    img, ppi = ds[0]
    assert isinstance(img, torch.Tensor)


def test_res_ds_transform_getter_setter(res_ds_root):
    import torchvision.transforms as T
    ds = ResDs.from_root(res_ds_root, GT_GLOB, image_crop="img")
    assert ds.input_transform is None
    ds.input_transform = T.ToTensor()
    assert ds.input_transform is not None


def test_res_ds_return_layout(res_ds_root):
    ds = ResDs.from_root(res_ds_root, GT_GLOB, image_crop="img", return_image_layout=True)
    img, ppi, layout = ds[0]
    assert isinstance(layout, dict)
    assert "class_names" in layout


def test_res_ds_random_split_sizes(res_ds_root):
    # Build a ds with multiple samples by creating more gt files
    import json, shutil
    from pathlib import Path
    root2 = Path(str(res_ds_root) + "_split")
    shutil.copytree(res_ds_root, root2, dirs_exist_ok=True)
    # Add 3 more charters
    for i in range(3):
        img_md5 = chr(ord("f") + i) * 32
        charter_dir = root2 / "RESARCH2" / ("c" * 32) / (chr(ord("d") + i) * 32)
        charter_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image as PILImage
        PILImage.new("RGB", (300, 400)).save(str(charter_dir / f"{img_md5}.img.jpg"))
        (charter_dir / f"{img_md5}.layout.pred.json").write_text(json.dumps({
            "class_names": ["No Class", "Ignore", "Img:CalibrationCard", "Img:Seal", "Img:WritableArea", "Wr:OldText"],
            "image_wh": [300, 400], "rect_LTRB": [[10, 20, 290, 380]], "rect_classes": [4],
        }))
        crops_dir = charter_dir / f"{img_md5}.layout.crops"
        crops_dir.mkdir(exist_ok=True)
        (crops_dir / "0.resolution.gt.json").write_text(json.dumps({
            "image_wh": [280, 360], "rect_LTRB": [[130, 10, 134, 170]],
            "rect_classes": [0], "class_names": ["5cm", "1cm", "5in", "1in", "Undefined"],
            "rect_captions": [""], "img_md5": "z" * 32,
        }))
    ds = ResDs.from_root(root2, GT_GLOB, image_crop="img")
    assert len(ds) == 4
    a, b = ds.random_split(0.75, seed=42)
    assert len(a) == 3
    assert len(b) == 1
    assert len(a) + len(b) == len(ds)


def test_res_ds_random_split_reproducible(res_ds_root):
    import json, shutil
    from pathlib import Path
    from PIL import Image as PILImage
    root3 = Path(str(res_ds_root) + "_repro")
    shutil.copytree(res_ds_root, root3, dirs_exist_ok=True)
    for i in range(5):
        img_md5 = chr(ord("g") + i) * 32
        cd = root3 / "REPRO" / ("h" * 32) / (chr(ord("i") + i) * 32)
        cd.mkdir(parents=True, exist_ok=True)
        PILImage.new("RGB", (300, 400)).save(str(cd / f"{img_md5}.img.jpg"))
        (cd / f"{img_md5}.layout.pred.json").write_text(json.dumps({
            "class_names": ["No Class", "Ignore", "Img:CalibrationCard", "Img:Seal", "Img:WritableArea", "Wr:OldText"],
            "image_wh": [300, 400], "rect_LTRB": [[10, 20, 290, 380]], "rect_classes": [4],
        }))
        cr = cd / f"{img_md5}.layout.crops"
        cr.mkdir(exist_ok=True)
        (cr / "0.resolution.gt.json").write_text(json.dumps({
            "image_wh": [280, 360], "rect_LTRB": [[130, 10, 134, 170]],
            "rect_classes": [0], "class_names": ["5cm", "1cm", "5in", "1in", "Undefined"],
            "rect_captions": [""], "img_md5": "z" * 32,
        }))
    ds = ResDs.from_root(root3, GT_GLOB, image_crop="img")
    a1, b1 = ds.random_split(0.5, seed=7)
    a2, b2 = ds.random_split(0.5, seed=7)
    assert [s[0] for s in a1.samples] == [s[0] for s in a2.samples]


# --- transform tests ---

def test_make_train_transform_no_crop():
    from ddp_resolution.res_ds import make_train_transform
    import torch
    t = make_train_transform()
    img = Image.new("RGB", (300, 400))
    result = t(img)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 400, 300)


def test_make_train_transform_with_patch():
    from ddp_resolution.res_ds import make_train_transform
    import torch
    t = make_train_transform(patch_size=224)
    img = Image.new("RGB", (300, 400))
    result = t(img)
    assert result.shape == (3, 224, 224)


def test_make_train_transform_small_image_padded():
    from ddp_resolution.res_ds import make_train_transform
    import torch
    t = make_train_transform(patch_size=256)
    img = Image.new("RGB", (100, 80))
    result = t(img)
    assert result.shape == (3, 256, 256)


def test_make_inference_transform_no_crop():
    from ddp_resolution.res_ds import make_inference_transform
    import torch
    t = make_inference_transform()
    img = Image.new("RGB", (300, 400))
    result = t(img)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 400, 300)


def test_make_inference_transform_with_patch():
    from ddp_resolution.res_ds import make_inference_transform
    import torch
    t = make_inference_transform(patch_size=224)
    img = Image.new("RGB", (300, 400))
    result = t(img)
    assert result.shape == (3, 224, 224)


def test_train_transform_normalised():
    from ddp_resolution.res_ds import make_train_transform, IMAGENET_MEAN, IMAGENET_STD
    import torch
    # Black image: after ImageNet normalisation all channels become negative
    t = make_train_transform(patch_size=4)
    img = Image.new("RGB", (10, 10), (0, 0, 0))
    result = t(img)
    assert result.max() < 0


def test_res_ds_from_gt_paths_list(res_ds_root):
    import glob as glob_module, os
    gt_paths = sorted(glob_module.glob(os.path.join(res_ds_root, GT_GLOB), recursive=True))
    ds = ResDs(gt_paths, image_crop="img")
    assert len(ds) == 1


def test_res_ds_from_root_matches_direct(res_ds_root):
    import glob as glob_module, os
    gt_paths = sorted(glob_module.glob(os.path.join(res_ds_root, GT_GLOB), recursive=True))
    ds_direct = ResDs(gt_paths, image_crop="img")
    ds_from_root = ResDs.from_root(res_ds_root, GT_GLOB, image_crop="img")
    assert len(ds_direct) == len(ds_from_root)
    assert ds_direct.samples == ds_from_root.samples


def test_res_ds_dominant_area_ratio_param(res_ds_root):
    # ratio=1.0 means no single rect can qualify (area/total <= 1.0 is always true)
    ds = ResDs.from_root(res_ds_root, GT_GLOB, image_crop="Img:WritableArea", dominant_area_ratio=1.0)
    assert len(ds) == 0


def test_res_ds_dominant_area_ratio_default(res_ds_root):
    ds = ResDs.from_root(res_ds_root, GT_GLOB, image_crop="Img:WritableArea")
    assert ds.dominant_area_ratio == ResDs.DOMINANT_AREA_RATIO


def test_res_ds_dominant_area_ratio_preserved_in_split(res_ds_root):
    import glob as glob_module, os
    gt_paths = sorted(glob_module.glob(os.path.join(res_ds_root, GT_GLOB), recursive=True))
    ds = ResDs(gt_paths, image_crop="img", dominant_area_ratio=0.7)
    a, b = ds.random_split(1.0)
    assert a.dominant_area_ratio == 0.7


# --- evaluate() tests ---

@pytest.fixture()
def eval_root(tmp_path):
    """Function-scoped copy of res_ds_root for evaluate tests that write pred files."""
    import json, shutil
    from PIL import Image as PILImage
    img_md5 = "d" * 32
    charter_dir = tmp_path / "RESARCH" / ("a" * 32) / ("b" * 32)
    charter_dir.mkdir(parents=True)
    PILImage.new("RGB", (300, 400), color=(200, 190, 180)).save(str(charter_dir / f"{img_md5}.img.jpg"))
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
    (crops_dir / "0.resolution.gt.json").write_text(json.dumps({
        "image_wh": [280, 360],
        "rect_LTRB": [[130, 10, 134, 170]],
        "rect_classes": [0],
        "class_names": ["5cm", "1cm", "5in", "1in", "Undefined"],
        "rect_captions": [""],
        "img_md5": "e" * 32,
    }))
    return tmp_path


def _eval_ds(eval_root):
    return ResDs.from_root(eval_root, "**/*.resolution.gt.json", image_crop="img")


def _pred_path(eval_root, ext=".res.pred.json"):
    img_md5 = "d" * 32
    return eval_root / "RESARCH" / ("a" * 32) / ("b" * 32) / f"{img_md5}{ext}"


def test_evaluate_perfect_prediction(eval_root):
    import json
    ds = _eval_ds(eval_root)
    gt_ppi = ds.samples[0][2]
    _pred_path(eval_root).write_text(json.dumps({"ppi": gt_ppi, "confidence": 1.0}))
    result = ds.evaluate()
    assert result["n_total"] == 1
    assert result["n_predicted"] == 1
    assert result["coverage"] == 1.0
    assert result["mean_gt_ppi"] is not None and result["mean_gt_ppi"] > 0
    assert abs(result["mae"]) < 1e-6
    assert abs(result["mape"]) < 1e-6
    assert abs(result["rmse"]) < 1e-6
    assert abs(result["median_ae"]) < 1e-6
    assert abs(result["log2_mae"]) < 1e-6
    assert abs(result["log2_rmse"]) < 1e-6


def test_evaluate_known_error(eval_root):
    import json
    ds = _eval_ds(eval_root)
    gt_ppi = ds.samples[0][2]
    _pred_path(eval_root).write_text(json.dumps({"ppi": gt_ppi + 30.0, "confidence": 0.5}))
    result = ds.evaluate()
    assert abs(result["mae"] - 30.0) < 1e-6
    assert abs(result["rmse"] - 30.0) < 1e-6
    assert abs(result["median_ae"] - 30.0) < 1e-6
    assert abs(result["mape"] - (30.0 / gt_ppi * 100.0)) < 1e-4


def test_evaluate_no_predictions(eval_root):
    ds = _eval_ds(eval_root)
    result = ds.evaluate()
    assert result["n_total"] == 1
    assert result["n_predicted"] == 0
    assert result["coverage"] == 0.0
    assert result["mean_gt_ppi"] is not None and result["mean_gt_ppi"] > 0
    assert result["mae"] is None
    assert result["mape"] is None
    assert result["rmse"] is None
    assert result["median_ae"] is None
    assert result["log2_mae"] is None
    assert result["log2_rmse"] is None


def test_evaluate_custom_ext(eval_root):
    import json
    ds = _eval_ds(eval_root)
    gt_ppi = ds.samples[0][2]
    _pred_path(eval_root, ".mymethod.json").write_text(json.dumps({"ppi": gt_ppi, "confidence": 0.9}))
    assert ds.evaluate(".res.pred.json")["n_predicted"] == 0
    assert ds.evaluate(".mymethod.json")["n_predicted"] == 1


def test_evaluate_log2_error_doubling(eval_root):
    import json, math
    ds = _eval_ds(eval_root)
    gt_ppi = ds.samples[0][2]
    _pred_path(eval_root).write_text(json.dumps({"ppi": gt_ppi * 2.0, "confidence": 0.5}))
    result = ds.evaluate()
    assert abs(result["log2_mae"] - 1.0) < 1e-6
    assert abs(result["log2_rmse"] - 1.0) < 1e-6
    # log errors are symmetric: halving gives same magnitude as doubling
    _pred_path(eval_root).write_text(json.dumps({"ppi": gt_ppi / 2.0, "confidence": 0.5}))
    result2 = ds.evaluate()
    assert abs(result2["log2_mae"] - 1.0) < 1e-6


def test_main_res_evaluate_runs(eval_root, capsys):
    import json
    from ddp_resolution.res_ds import main_res_evaluate
    ds = _eval_ds(eval_root)
    gt_ppi = ds.samples[0][2]
    _pred_path(eval_root).write_text(json.dumps({"ppi": gt_ppi, "confidence": 1.0}))

    import sys
    old_argv = sys.argv
    sys.argv = ["ddp_res_evaluate", "-fsdb_roots", str(eval_root), "-pred_exts", ".res.pred.json,.other.json"]
    try:
        main_res_evaluate()
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.strip()]
    assert lines[0].startswith("ext")
    assert "mean_gt_ppi" in lines[0]
    assert any(".res.pred.json" in l for l in lines)
    assert any(".other.json" in l for l in lines)
    # .res.pred.json should show n_predicted=1, .other.json should show 0
    pred_line = next(l for l in lines if ".res.pred.json" in l)
    other_line = next(l for l in lines if ".other.json" in l)
    assert "1" in pred_line
    assert "N/A" in other_line


def test_evaluate_mean_gt_ppi(eval_root):
    ds = _eval_ds(eval_root)
    gt_ppi = ds.samples[0][2]
    result = ds.evaluate()
    assert abs(result["mean_gt_ppi"] - gt_ppi) < 1e-6


def test_main_res_evaluate_plot_saved(eval_root, tmp_path):
    import json, sys
    from ddp_resolution.res_ds import main_res_evaluate
    ds = _eval_ds(eval_root)
    gt_ppi = ds.samples[0][2]
    _pred_path(eval_root).write_text(json.dumps({"ppi": gt_ppi, "confidence": 1.0}))
    plotfname = str(tmp_path / "out.png")
    old_argv = sys.argv
    sys.argv = [
        "ddp_res_evaluate",
        "-fsdb_roots", str(eval_root),
        "-pred_exts", ".res.pred.json",
        "-plot",
        "-plotfname", plotfname,
    ]
    import matplotlib
    matplotlib.use("Agg")
    try:
        main_res_evaluate()
    finally:
        sys.argv = old_argv
    import os
    assert os.path.exists(plotfname)
