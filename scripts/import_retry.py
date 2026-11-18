"""Import the fine-tune retry (attempt 2) from a Kaggle zip that used attempt 1's output names.

The Kaggle notebook that ran attempt 2 still had attempt 1's cells, so its zip names the retry
`plantcad_adapted` / `plantcad_ft` / `plantcad_ft_head`, the same names attempt 1 used. Unzipping it
straight into work/ would overwrite attempt 1's record. This script:

1. checks the zip really is attempt 2 (its finetune_log.json config must say 3 windows per read and
   backbone lr 1e-4), and refuses otherwise;
2. copies only the retry's outputs, under attempt 2's names:
     features/plantcad_ft      -> features/plantcad_ft2
     models/plantcad_adapted   -> models/plantcad_adapted_r2
     models/plantcad_ft_head   -> models/plantcad_ft2_head
   rewriting the name fields inside their metadata, and refuses to overwrite anything;
3. leaves the zip's re-made frozen embeddings and frozen head alone (work/ already has attempt 1's),
   but compares the re-made frozen embeddings with the ones in work/ as a reproducibility check.

Usage:  python scripts/import_retry.py /path/to/gpu_outputs.zip [--work work]
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

RENAMES = {
    "features/plantcad_ft": "features/plantcad_ft2",
    "models/plantcad_adapted": "models/plantcad_adapted_r2",
    "models/plantcad_ft_head": "models/plantcad_ft2_head",
}
EXPECTED = {"windows_per_read": 3, "lr_backbone": 1.0e-4}


def check_attempt_2(zf: zipfile.ZipFile) -> dict:
    name = "models/plantcad_adapted/finetune_log.json"
    if name not in zf.namelist():
        sys.exit(f"{name} is not in the zip: is this the Kaggle output?")
    log = json.loads(zf.read(name))
    cfg = log["config"]
    for key, want in EXPECTED.items():
        if float(cfg.get(key, -1)) != float(want):
            sys.exit(f"this zip is not attempt 2: finetune {key} = {cfg.get(key)} (attempt 2 used {want})")
    return log


NAME_MAP = {"plantcad_ft": "plantcad_ft2", "plantcad_adapted": "plantcad_adapted_r2", "plantcad_ft_head": "plantcad_ft2_head"}


def _rename(value):
    if isinstance(value, str):
        if value in NAME_MAP:
            return NAME_MAP[value]
        return value.replace("models/plantcad_adapted/", "models/plantcad_adapted_r2/")
    return value


def rewrite_metadata(dest: Path, old: str) -> None:
    for p in dest.glob("*.json"):
        d = json.loads(p.read_text())
        d = {k: (_rename(v) if k in ("name", "out", "head", "features", "checkpoint") else v) for k, v in d.items()}
        d["renamed_from"] = old
        p.write_text(json.dumps(d, indent=2))
    head = dest / "head.pt"
    if head.exists():
        import torch

        state = torch.load(head, map_location="cpu", weights_only=False)
        state["features"] = _rename(state.get("features"))
        state["renamed_from"] = old
        torch.save(state, head)


def compare_frozen(zf: zipfile.ZipFile, work: Path) -> list[str]:
    lines = []
    for split in ("train", "val", "test"):
        name = f"features/plantcad/{split}.npy"
        mine = work / name
        if name not in zf.namelist() or not mine.exists():
            continue
        a = np.load(io.BytesIO(zf.read(name)))
        b = np.load(mine)
        if a.shape != b.shape:
            lines.append(f"{split}: shapes differ {a.shape} vs {b.shape}")
            continue
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        lines.append(f"{split}: max |difference| {np.abs(a - b).max():.2e}, min cosine {cos.min():.6f}")
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip", type=Path)
    ap.add_argument("--work", type=Path, default=Path("work"))
    args = ap.parse_args(argv)
    with zipfile.ZipFile(args.zip) as zf:
        log = check_attempt_2(zf)
        for new in RENAMES.values():
            if (args.work / new).exists():
                sys.exit(f"{args.work / new} already exists; not overwriting")
        with tempfile.TemporaryDirectory() as tmp:
            members = [m for m in zf.namelist() if any(m.startswith(old + "/") for old in RENAMES)]
            for m in members:  # extract member by member: no path outside tmp is ever written
                target = Path(tmp) / m
                if not target.resolve().is_relative_to(Path(tmp).resolve()):
                    sys.exit(f"refusing suspicious path in zip: {m}")
                if not m.endswith("/"):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(m))
            for old, new in RENAMES.items():
                src, dest = Path(tmp) / old, args.work / new
                if not src.exists():
                    sys.exit(f"{old} is missing from the zip")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest)
                rewrite_metadata(dest, Path(old).name)
                print(f"[import] {old} -> {dest}")
        check = compare_frozen(zf, args.work)
    best = log.get("best", {})
    print(f"[import] attempt 2: best step {best.get('step')} score {best.get('score', float('nan')):.3f}, "
          f"{log.get('skipped_steps')} updates skipped, batch {log.get('effective_batch')}")
    if check:
        print("[import] frozen embeddings re-made on Kaggle vs the ones in work/ (should match closely):")
        for line in check:
            print(f"         {line}")


if __name__ == "__main__":
    main()
