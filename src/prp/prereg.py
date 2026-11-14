"""Preregistration lock: targets are written down and frozen before the test split is ever scored.

`prp freeze` hashes PREREG.md, the configs, the calibrated thresholds and the trained weights into
prereg.lock.json. `prp evaluate --split test` refuses to run without that lock, refuses if any frozen
file has changed, and refuses a second run unless a reason is given, which is recorded as a deviation.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

from .config import Panel
from .genomes import sha256_file
from .paths import Layout

PLACEHOLDER = "TBD"


class PreregError(RuntimeError):
    pass


def _git_commit(root: Path) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except OSError:
        return None


def frozen_files(root: Path, layout: Layout, panel_path: Path, exp_path: Path | None) -> list[Path]:
    files = [root / "PREREG.md", panel_path]
    if exp_path is not None:
        files.append(exp_path)
    files += sorted((layout.root / "calibration").glob("*.json"))
    files += sorted((layout.root / "models").glob("*/head.pt"))
    files += sorted((layout.root / "models").glob("*/encoder.pt"))
    files.append(layout.lock)
    return [f for f in files if f.exists()]


def freeze(root: Path, layout: Layout, panel: Panel, exp_path: Path | None) -> dict:
    prereg = root / "PREREG.md"
    if not prereg.exists():
        raise PreregError(f"{prereg} not found")
    text = prereg.read_text()
    if PLACEHOLDER in text:
        n = text.count(PLACEHOLDER)
        raise PreregError(f"PREREG.md still has {n} '{PLACEHOLDER}' placeholder(s); fill every target first")
    panel.validate(strict_pretrain=True)
    if not list((layout.root / "calibration").glob("*.json")):
        raise PreregError("no calibration files; run `prp calibrate` for each method first")
    if layout.prereg_lock.exists():
        raise PreregError(f"{layout.prereg_lock} already exists; the preregistration is frozen")
    files = frozen_files(root, layout, Path(panel.path), exp_path)
    lock = {
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(root),
        "files": {str(f.resolve().relative_to(root.resolve()) if f.resolve().is_relative_to(root.resolve()) else f): sha256_file(f) for f in files},
    }
    layout.prereg_lock.write_text(json.dumps(lock, indent=2))
    return lock


def check_lock(root: Path, layout: Layout) -> dict:
    if not layout.prereg_lock.exists():
        raise PreregError("test evaluation requires a frozen preregistration: run `prp freeze` first")
    lock = json.loads(layout.prereg_lock.read_text())
    changed = []
    for rel, digest in lock["files"].items():
        p = Path(rel) if Path(rel).is_absolute() else root / rel
        if not p.exists():
            changed.append(f"{rel} (missing)")
        elif sha256_file(p) != digest:
            changed.append(rel)
    if changed:
        raise PreregError("files changed since the preregistration was frozen: " + ", ".join(changed))
    return lock


def register_test_run(layout: Layout, methods: list[str], rerun_reason: str | None) -> dict:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if layout.test_marker.exists():
        rec = json.loads(layout.test_marker.read_text())
        if not rerun_reason:
            raise PreregError(
                f"the test split was already evaluated at {rec['first_run']}; re-running needs "
                "--rerun-reason, which is recorded as a deviation")
        rec.setdefault("deviations", []).append({"at": now, "reason": rerun_reason, "methods": methods})
    else:
        rec = {"first_run": now, "methods": methods, "lock_sha256": sha256_file(layout.prereg_lock), "deviations": []}
    layout.test_marker.write_text(json.dumps(rec, indent=2))
    return rec
