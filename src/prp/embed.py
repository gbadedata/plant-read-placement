"""Embed every window with an encoder and average windows into one vector per read."""

from __future__ import annotations

import json
import time

import numpy as np

from .paths import Layout
from .windows import load_reads, load_windows


def reads_from_windows(win_emb: np.ndarray, read_index: np.ndarray, n_reads: int) -> np.ndarray:
    out = np.zeros((n_reads, win_emb.shape[1]), dtype=np.float64)
    np.add.at(out, read_index, win_emb)
    counts = np.bincount(read_index, minlength=n_reads).astype(np.float64)
    out /= np.maximum(counts, 1)[:, None]
    return out.astype(np.float32)


def embed_split(encoder, layout: Layout, split: str, name: str, batch_size: int = 64, meta: dict | None = None) -> np.ndarray:
    windows, read_index = load_windows(layout, split)
    reads = load_reads(layout, split)
    t0 = time.time()
    chunks = []
    for i in range(0, len(windows), batch_size):
        chunks.append(np.asarray(encoder.embed_windows(windows[i : i + batch_size]), dtype=np.float32))
        done = i + batch_size
        if (i // batch_size) % 200 == 0:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"[embed] {name}/{split}: {min(done, len(windows))}/{len(windows)} windows ({rate:.0f}/s)", flush=True)
    win_emb = np.concatenate(chunks)
    feats = reads_from_windows(win_emb, read_index, len(reads))
    np.save(layout.features(name, split), feats)
    info = {"name": name, "split": split, "n_reads": len(reads), "n_windows": len(windows),
            "dim": int(feats.shape[1]), "seconds": round(time.time() - t0, 1), **(meta or {})}
    layout.features(name, split).with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"[embed] {name}/{split}: done, dim={feats.shape[1]}, {info['seconds']} s")
    return feats


def load_features(layout: Layout, name: str, split: str) -> np.ndarray:
    return np.load(layout.features(name, split))
