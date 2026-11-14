"""Metrics for the three-way placement decision, with species-level (cluster) bootstrap intervals.

Specimens from the same species are not independent, so confidence intervals resample *species*
within each truth class, not specimens. With a handful of held-out species per class the intervals
are wide; that is the honest answer at this scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import CLASSES
from .placement import balanced_accuracy, correct_mask, decide


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def point_metrics(scores: pd.DataFrame, t1: float, t2: float) -> dict:
    d1 = scores["d1"].to_numpy()
    truth = scores["truth"].to_numpy()
    pred = decide(d1, t1, t2)
    ok = correct_mask(scores, pred)
    out: dict = {"n_specimens": len(scores), "balanced_accuracy": balanced_accuracy(scores, pred)}
    for c in CLASSES:
        m = truth == c
        out[f"recall_{c}"] = float(ok[m].mean()) if m.any() else float("nan")
        out[f"n_{c}"] = int(m.sum())
    km = truth == "known"
    rm = truth == "relative"
    nm = truth == "new"
    out["known_species_top1"] = float((scores.nearest_species[km] == scores.species[km]).mean()) if km.any() else float("nan")
    out["relative_genus_top1"] = float((scores.nearest_genus[rm] == scores.genus[rm]).mean()) if rm.any() else float("nan")
    out["new_family_top1"] = float((scores.nearest_family[nm] == scores.family[nm]).mean()) if nm.any() else float("nan")
    out["auroc_new_vs_rest"] = _auc(nm.astype(int), d1)
    out["auroc_known_vs_rest"] = _auc(km.astype(int), -d1)
    conf = pd.crosstab(pd.Categorical(truth, CLASSES), pd.Categorical(pred, CLASSES), dropna=False)
    out["confusion"] = {str(r): {str(c): int(conf.loc[r, c]) for c in conf.columns} for r in conf.index}
    return out


def cluster_bootstrap(scores: pd.DataFrame, t1: float, t2: float, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    groups = {c: [g.index.to_numpy() for _, g in scores[scores.truth == c].groupby("species")] for c in CLASSES}
    groups = {c: v for c, v in groups.items() if v}
    ba, auc = [], []
    d1_all = scores["d1"].to_numpy()
    for _ in range(n_boot):
        idx = np.concatenate([np.concatenate([v[j] for j in rng.integers(0, len(v), len(v))]) for v in groups.values()])
        sub = scores.loc[idx]
        pred = decide(d1_all[idx], t1, t2)
        ba.append(balanced_accuracy(sub, pred))
        y = (sub.truth.to_numpy() == "new").astype(int)
        auc.append(_auc(y, d1_all[idx]))

    def ci(v):
        v = np.asarray([x for x in v if not np.isnan(x)])
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [float("nan")] * 2

    return {"balanced_accuracy_ci95": ci(ba), "auroc_new_vs_rest_ci95": ci(auc), "n_boot": n_boot,
            "species_per_class": {c: len(v) for c, v in groups.items()}}


def stratified(scores: pd.DataFrame, t1: float, t2: float, column: str, bins: list[float]) -> list[dict]:
    rows = []
    vals = scores[column].to_numpy()
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (vals >= lo) & (vals < hi)
        if not m.any():
            continue
        sub = scores[m]
        pred = decide(sub.d1.to_numpy(), t1, t2)
        ok = correct_mask(sub, pred)
        row = {"bin": f"[{lo:g}, {hi:g})", "n": int(m.sum()), "balanced_accuracy": balanced_accuracy(sub, pred)}
        for c in CLASSES:
            mc = sub.truth.to_numpy() == c
            row[f"recall_{c}"] = float(ok[mc].mean()) if mc.any() else float("nan")
        rows.append(row)
    return rows


def evaluate(scores: pd.DataFrame, calib: dict, exp, seed: int) -> dict:
    ev = exp["evaluation"]
    scores = scores.reset_index(drop=True)
    t1, t2 = calib["t1"], calib["t2"]
    res = point_metrics(scores, t1, t2)
    res.update(cluster_bootstrap(scores, t1, t2, int(ev["bootstrap"]), seed))
    res["by_read_identity"] = stratified(scores, t1, t2, "mean_identity", list(ev["identity_bins"]))
    res["by_read_length"] = stratified(scores, t1, t2, "mean_length", list(ev["length_bins"]))
    res["thresholds"] = {"t1": t1, "t2": t2}
    return res
