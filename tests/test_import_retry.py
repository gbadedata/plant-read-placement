"""The retry importer must keep attempt 1 intact and only accept attempt 2's zip."""
from __future__ import annotations

import json
import runpy
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_retry.py"


def _zip(path: Path, windows_per_read: int, lr: float, frozen: dict[str, np.ndarray], prefix: str = "") -> Path:
    import io

    def npy(a):
        b = io.BytesIO()
        np.save(b, a)
        return b.getvalue()

    head = io.BytesIO()
    torch.save({"config": {}, "state_dict": {}, "features": "plantcad_ft", "step": 5}, head)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(prefix + "models/plantcad_adapted/finetune_log.json", json.dumps(
            {"out": "plantcad_adapted", "config": {"windows_per_read": windows_per_read, "lr_backbone": lr},
             "best": {"step": 500, "score": 0.3}, "skipped_steps": 3, "effective_batch": {}}))
        zf.writestr(prefix + "models/plantcad_adapted/encoder.pt", b"x")
        zf.writestr(prefix + "models/plantcad_ft_head/head.pt", head.getvalue())
        zf.writestr(prefix + "models/plantcad_ft_head/train_log.json", json.dumps({"features": "plantcad_ft", "head": "plantcad_ft_head"}))
        zf.writestr(prefix + "models/plantcad_head/head.pt", b"retrained frozen head: must not be imported")
        for split, a in frozen.items():
            zf.writestr(prefix + f"features/plantcad/{split}.npy", npy(a))
            zf.writestr(prefix + f"features/plantcad_ft/{split}.npy", npy(a + 1))
            zf.writestr(prefix + f"features/plantcad_ft/{split}.json", json.dumps(
                {"name": "plantcad_ft", "checkpoint": "work/models/plantcad_adapted/encoder.pt"}))
    return path


def _run(*args):
    mod = runpy.run_path(str(SCRIPT))
    mod["main"](list(args))


def test_import_renames_and_protects_attempt_1(tmp_path, capsys):
    rng = np.random.default_rng(0)
    work = tmp_path / "work"
    frozen = {s: rng.normal(size=(5, 4)).astype(np.float32) for s in ("train", "val", "test")}
    for s, a in frozen.items():
        (work / "features" / "plantcad").mkdir(parents=True, exist_ok=True)
        np.save(work / "features" / "plantcad" / f"{s}.npy", a)
    (work / "models" / "plantcad_ft_head").mkdir(parents=True)
    (work / "models" / "plantcad_ft_head" / "marker").write_text("attempt 1")

    _run(str(_zip(tmp_path / "r2.zip", 3, 1e-4, frozen)), "--work", str(work))
    assert (work / "models" / "plantcad_ft_head" / "marker").read_text() == "attempt 1"
    assert not (work / "models" / "plantcad_head").exists()
    meta = json.loads((work / "features" / "plantcad_ft2" / "val.json").read_text())
    assert meta["name"] == "plantcad_ft2" and meta["checkpoint"] == "work/models/plantcad_adapted_r2/encoder.pt"
    assert json.loads((work / "models" / "plantcad_adapted_r2" / "finetune_log.json").read_text())["out"] == "plantcad_adapted_r2"
    log = json.loads((work / "models" / "plantcad_ft2_head" / "train_log.json").read_text())
    assert (log["features"], log["head"]) == ("plantcad_ft2", "plantcad_ft2_head")
    assert torch.load(work / "models" / "plantcad_ft2_head" / "head.pt", weights_only=False)["features"] == "plantcad_ft2"
    assert "min cosine 1.000000" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="already exists"):
        _run(str(tmp_path / "r2.zip"), "--work", str(work))


def test_import_refuses_attempt_1_zip(tmp_path):
    with pytest.raises(SystemExit, match="not attempt 2"):
        _run(str(_zip(tmp_path / "r1.zip", 1, 2e-5, {})), "--work", str(tmp_path / "work"))


def test_import_accepts_kaggle_download_all_layout(tmp_path):
    work = tmp_path / "work"
    frozen = {"val": np.ones((3, 4), dtype=np.float32)}
    _run(str(_zip(tmp_path / "all.zip", 3, 1e-4, frozen, prefix="prp/work/")), "--work", str(work))
    assert (work / "features" / "plantcad_ft2" / "val.npy").exists()
    assert (work / "models" / "plantcad_adapted_r2" / "finetune_log.json").exists()
    assert (work / "models" / "plantcad_ft2_head" / "head.pt").exists()


def test_import_accepts_a_downloaded_folder(tmp_path):
    archive = _zip(tmp_path / "all.zip", 3, 1e-4, {"val": np.ones((3, 4), dtype=np.float32)}, prefix="prp/work/")
    folder = tmp_path / "kaggle_r2"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(folder)
    work = tmp_path / "work"
    _run(str(folder), "--work", str(work))
    assert (work / "features" / "plantcad_ft2" / "val.npy").exists()
    assert (work / "models" / "plantcad_ft2_head" / "train_log.json").exists()
