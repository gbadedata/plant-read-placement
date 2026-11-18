"""Cut each read into fixed-size windows (PlantCaduceus takes 512 bp) and build the per-split read table.

Windows are evenly spaced along the read, up to `max_per_read`, so long reads contribute more of
their sequence without letting any one read dominate. The read table is the label source for every
later step: species, genus, family, role and truth class (known / relative / new).
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from .config import Experiment, Panel
from .paths import Layout
from .seqio import encode, read_fastq
from .simulate import is_artifact, parse_identity

READ_COLUMNS = ["read_idx", "read_id", "species", "genus", "family", "order", "role", "truth", "split",
                "length", "identity", "artifact", "n_windows", "first_window"]


def window_starts(length: int, size: int, max_n: int) -> np.ndarray:
    n = min(max_n, length // size)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.array([(length - size) // 2], dtype=np.int64)
    return np.linspace(0, length - size, n).round().astype(np.int64)


def build_windows(panel: Panel, exp: Experiment, layout: Layout, splits=("train", "val", "test")) -> dict:
    size = int(exp["windows"]["size"])
    max_n = int(exp["windows"]["max_per_read"])
    summary = {}
    for split in splits:
        wins: list[np.ndarray] = []
        rows: list[dict] = []
        w_total = 0
        for s in panel.species:
            if split not in s.splits:
                continue
            path = layout.reads(s.id, split)
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run `prp simulate`")
            for header, seq, _ in read_fastq(path):
                read_id, _, desc = header.partition(" ")
                starts = window_starts(len(seq), size, max_n)
                if len(starts) == 0:
                    continue
                codes = encode(seq)
                block = np.stack([codes[a : a + size] for a in starts])
                wins.append(block)
                rows.append({
                    "read_idx": len(rows), "read_id": read_id, "species": s.id, "genus": s.genus,
                    "family": s.family, "order": s.order, "role": s.role, "truth": s.truth, "split": split,
                    "length": len(seq), "identity": parse_identity(desc), "artifact": int(is_artifact(desc)),
                    "n_windows": len(starts), "first_window": w_total,
                })
                w_total += len(starts)
        if not rows:
            continue
        arr = np.concatenate(wins).astype(np.uint8)
        read_index = np.repeat(np.arange(len(rows), dtype=np.int32), [r["n_windows"] for r in rows])
        np.savez_compressed(layout.windows(split), windows=arr, read_index=read_index)
        with open(layout.read_table(split), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=READ_COLUMNS, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        summary[split] = {"reads": len(rows), "windows": int(arr.shape[0])}
        print(f"[windows] {split}: {len(rows)} reads, {arr.shape[0]} windows of {size} bp")
    return summary


def load_reads(layout: Layout, split: str) -> pd.DataFrame:
    return pd.read_csv(layout.read_table(split), sep="\t")


def load_windows(layout: Layout, split: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(layout.windows(split))
    return z["windows"], z["read_index"]


def windows_of_read(windows: np.ndarray, reads: pd.DataFrame, i: int) -> np.ndarray:
    r = reads.iloc[i]
    return windows[int(r.first_window) : int(r.first_window) + int(r.n_windows)]


__all__ = ["build_windows", "load_reads", "load_windows", "window_starts", "windows_of_read"]
