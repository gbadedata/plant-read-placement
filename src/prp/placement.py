"""Place specimens against the reference panel and decide: known species, nearest relative, or likely new.

A specimen is a set of n reads from one species (n=1 is a single read). Its embedding is the
normalised mean of its read embeddings. Distance to each panel species is 1 - cosine similarity to
that species' prototype (normalised mean of its training-region reads). With d1 the distance to the
nearest prototype and two thresholds t1 <= t2 fitted on validation only:

    d1 <= t1        -> known species   (report the nearest species)
    t1 < d1 <= t2   -> nearest relative (report the nearest species' genus)
    d1 > t2         -> likely new

The same rule and the same calibration are applied to every method, including minimap2, whose
"distance" is 1 - mean alignment identity of the specimen's reads (unmapped reads count as 0).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .config import CLASSES
from .heads import ProjectionHead
from .paths import Layout
from .simulate import derive_seed


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class RawTransform:
    """z-score with training-split statistics, then L2-normalise (for methods without a trained head)."""

    def __init__(self, train_feats: np.ndarray):
        self.mean = train_feats.mean(axis=0)
        self.std = np.maximum(train_feats.std(axis=0), 1e-6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return l2n((x - self.mean) / self.std)


class HeadTransform:
    def __init__(self, head: ProjectionHead, batch: int = 4096):
        self.head = head.eval()
        self.batch = batch

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        outs = [self.head(torch.from_numpy(x[i : i + self.batch]).float()).numpy() for i in range(0, len(x), self.batch)]
        return np.concatenate(outs) if outs else np.zeros((0, self.head.out_dim), dtype=np.float32)


def load_head(layout: Layout, name: str) -> ProjectionHead:
    state = torch.load(layout.model_dir(name) / "head.pt", map_location="cpu")
    return ProjectionHead.from_state(state)


def prototypes(train_emb: np.ndarray, train_reads: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    keep = train_reads["artifact"].to_numpy() == 0
    reads = train_reads[keep]
    emb = train_emb[keep]
    meta = reads.groupby("species", sort=True).first()[["genus", "family"]].reset_index()
    protos = np.stack([emb[(reads["species"] == sp).to_numpy()].mean(axis=0) for sp in meta["species"]])
    return l2n(protos), meta


def build_specimens(reads: pd.DataFrame, n: int, max_per_species: int, seed: int) -> list[tuple[str, np.ndarray]]:
    specimens = []
    for sp, grp in reads.groupby("species", sort=True):
        idx = grp["read_idx"].to_numpy()
        rng = np.random.default_rng(derive_seed(seed, sp, n))
        idx = rng.permutation(idx)
        n_spec = min(len(idx) // n, max_per_species)
        for j in range(n_spec):
            specimens.append((f"{sp}.n{n}.{j:04d}", idx[j * n : (j + 1) * n]))
    return specimens


def score_embeddings(emb: np.ndarray, reads: pd.DataFrame, protos: np.ndarray, proto_meta: pd.DataFrame,
                     n: int, max_per_species: int, seed: int) -> pd.DataFrame:
    """One row per specimen: truth labels, d1, d2, nearest species/genus/family."""
    reads = reads.set_index("read_idx", drop=False)
    specimens = build_specimens(reads.reset_index(drop=True), n, max_per_species, seed)
    if not specimens:
        return pd.DataFrame()
    spec_emb = l2n(np.stack([emb[ix].mean(axis=0) for _, ix in specimens]))
    dist = 1.0 - spec_emb @ protos.T
    order = np.argsort(dist, axis=1)
    rows = []
    for k, (sid, ix) in enumerate(specimens):
        first = reads.loc[ix[0]]
        j1 = order[k, 0]
        j2 = order[k, 1] if dist.shape[1] > 1 else j1
        rows.append({
            "specimen": sid, "n": n, "species": first.species, "genus": first.genus, "family": first.family,
            "truth": first.truth, "d1": float(dist[k, j1]), "d2": float(dist[k, j2]),
            "nearest_species": proto_meta.species.iloc[j1], "nearest_genus": proto_meta.genus.iloc[j1],
            "nearest_family": proto_meta.family.iloc[j1],
            "mean_identity": float(np.nanmean(reads.loc[ix, "identity"])) if reads.loc[ix, "identity"].notna().any() else float("nan"),
            "mean_length": float(reads.loc[ix, "length"].mean()),
            "artifact_frac": float(reads.loc[ix, "artifact"].mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- decision and calibration

def decide(d1: np.ndarray, t1: float, t2: float) -> np.ndarray:
    return np.where(d1 <= t1, "known", np.where(d1 <= t2, "relative", "new"))


def correct_mask(scores: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """End-to-end correctness: known needs the right species, relative the right genus, new just 'new'."""
    truth = scores["truth"].to_numpy()
    ok_known = (truth == "known") & (pred == "known") & (scores["nearest_species"].to_numpy() == scores["species"].to_numpy())
    ok_rel = (truth == "relative") & (pred == "relative") & (scores["nearest_genus"].to_numpy() == scores["genus"].to_numpy())
    ok_new = (truth == "new") & (pred == "new")
    return ok_known | ok_rel | ok_new


def balanced_accuracy(scores: pd.DataFrame, pred: np.ndarray) -> float:
    ok = correct_mask(scores, pred)
    truth = scores["truth"].to_numpy()
    accs = [ok[truth == c].mean() for c in CLASSES if (truth == c).any()]
    return float(np.mean(accs)) if accs else float("nan")


def calibrate(scores: pd.DataFrame, grid: int = 200) -> dict:
    """Choose t1 <= t2 on validation scores to maximise end-to-end balanced accuracy."""
    d1 = scores["d1"].to_numpy()
    cands = np.unique(np.quantile(d1, np.linspace(0, 1, grid)))
    cands = np.concatenate([[d1.min() - 1e-9], cands, [d1.max() + 1e-9]])
    truth = scores["truth"].to_numpy()
    sp_ok = scores["nearest_species"].to_numpy() == scores["species"].to_numpy()
    ge_ok = scores["nearest_genus"].to_numpy() == scores["genus"].to_numpy()
    masks = {c: truth == c for c in CLASSES}
    best = (-1.0, cands[0], cands[0])
    for i, t1 in enumerate(cands):
        is_known = d1 <= t1
        for t2 in cands[i:]:
            is_rel = (~is_known) & (d1 <= t2)
            is_new = d1 > t2
            accs = []
            if masks["known"].any():
                accs.append((is_known & sp_ok)[masks["known"]].mean())
            if masks["relative"].any():
                accs.append((is_rel & ge_ok)[masks["relative"]].mean())
            if masks["new"].any():
                accs.append(is_new[masks["new"]].mean())
            ba = float(np.mean(accs))
            if ba > best[0] + 1e-12:
                best = (ba, float(t1), float(t2))
    return {"t1": best[1], "t2": best[2], "val_balanced_accuracy": best[0], "n_val_specimens": len(scores)}


def save_calibration(layout: Layout, method: str, calib: dict) -> None:
    layout.calibration(method).write_text(json.dumps(calib, indent=2))


def load_calibration(layout: Layout, method: str) -> dict:
    return json.loads(layout.calibration(method).read_text())
