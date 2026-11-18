from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prp.metrics import cluster_bootstrap, point_metrics
from prp.paths import Layout
from prp.placement import (
    balanced_accuracy,
    build_specimens,
    calibrate,
    correct_mask,
    decide,
    prototypes,
    score_embeddings,
)
from prp.prereg import PreregError, check_lock, freeze, register_test_run


def _scores(rows):
    return pd.DataFrame(rows, columns=["species", "genus", "family", "truth", "d1", "nearest_species",
                                       "nearest_genus", "nearest_family"])


def test_decide_bands():
    assert list(decide(np.array([0.05, 0.2, 0.5]), 0.1, 0.3)) == ["known", "relative", "new"]


def test_correct_mask_requires_right_label():
    s = _scores([
        ["a", "A", "F", "known", 0.0, "a", "A", "F"],     # right species
        ["a", "A", "F", "known", 0.0, "b", "A", "F"],     # wrong species
        ["c", "A", "F", "relative", 0.2, "a", "A", "F"],  # right genus
        ["c", "A", "F", "relative", 0.2, "x", "X", "F"],  # wrong genus
        ["z", "Z", "G", "new", 0.9, "a", "A", "F"],
    ])
    pred = np.array(["known", "known", "relative", "relative", "new"])
    assert correct_mask(s, pred).tolist() == [True, False, True, False, True]
    assert balanced_accuracy(s, pred) == pytest.approx((0.5 + 0.5 + 1) / 3)


def test_calibrate_recovers_separable_thresholds():
    rng = np.random.default_rng(0)
    rows = []
    for truth, lo, hi in [("known", 0.0, 0.1), ("relative", 0.2, 0.3), ("new", 0.5, 0.7)]:
        for _ in range(50):
            rows.append(["s", "G", "F", truth, rng.uniform(lo, hi), "s", "G", "F"])
    s = _scores(rows)
    c = calibrate(s)
    assert c["val_balanced_accuracy"] == pytest.approx(1.0)
    assert 0.1 <= c["t1"] < 0.2 and 0.3 <= c["t2"] < 0.5


def test_specimens_disjoint_and_deterministic():
    reads = pd.DataFrame({"read_idx": np.arange(23), "species": ["a"] * 13 + ["b"] * 10})
    s1 = build_specimens(reads, 5, 100, seed=1)
    s2 = build_specimens(reads, 5, 100, seed=1)
    assert [x[0] for x in s1] == [x[0] for x in s2] and len(s1) == 2 + 2
    for _, ix in s1:
        assert len(set(ix)) == 5
    a_reads = np.concatenate([ix for sid, ix in s1 if sid.startswith("a")])
    assert len(set(a_reads)) == len(a_reads)


def test_prototype_scoring_end_to_end():
    rng = np.random.default_rng(0)
    centers = {"a": np.array([1.0, 0, 0]), "b": np.array([0, 1.0, 0])}
    tr = pd.DataFrame({"read_idx": np.arange(40), "species": ["a"] * 20 + ["b"] * 20,
                       "genus": ["A"] * 20 + ["B"] * 20, "family": ["F"] * 40, "artifact": 0})
    X = np.stack([centers[s] + 0.05 * rng.standard_normal(3) for s in tr.species])
    protos, meta = prototypes(X, tr)
    q = pd.DataFrame({"read_idx": np.arange(10), "species": ["a"] * 5 + ["c"] * 5, "genus": ["A"] * 10,
                      "family": ["F"] * 10, "truth": ["known"] * 5 + ["relative"] * 5,
                      "identity": 0.9, "length": 5000, "artifact": 0})
    Q = np.stack([centers["a"] + 0.05 * rng.standard_normal(3) for _ in range(5)] +
                 [np.array([0.7, 0.0, 0.7]) for _ in range(5)])
    sc = score_embeddings(Q, q, protos, meta, 1, 100, seed=0)
    assert (sc[sc.truth == "known"].nearest_species == "a").all()
    assert sc[sc.truth == "known"].d1.max() < sc[sc.truth == "relative"].d1.min()
    m = point_metrics(sc, *[calibrate(sc)[k] for k in ("t1", "t2")])
    assert m["recall_known"] == 1.0 and m["recall_relative"] == 1.0


def test_cluster_bootstrap_brackets_point_estimate():
    rng = np.random.default_rng(3)
    rows = []
    for truth, mu in [("known", 0.1), ("relative", 0.3), ("new", 0.6)]:
        for sp in range(4):
            for _ in range(20):
                rows.append([f"{truth}{sp}", "G", "F", truth, rng.normal(mu, 0.08), f"{truth}{sp}", "G", "F"])
    s = _scores(rows)
    c = calibrate(s)
    pm = point_metrics(s, c["t1"], c["t2"])
    bs = cluster_bootstrap(s, c["t1"], c["t2"], 300, seed=0)
    lo, hi = bs["balanced_accuracy_ci95"]
    assert lo <= pm["balanced_accuracy"] <= hi and bs["species_per_class"] == {"known": 4, "relative": 4, "new": 4}


def test_prereg_freeze_and_guards(tmp_path, panel):
    root = tmp_path
    layout = Layout(root / "work", root)
    (root / "configs").mkdir()
    panel_file = root / "configs" / "panel.yaml"
    panel_file.write_text("x")
    panel.path = panel_file
    (root / "PREREG.md").write_text("target: TBD\n")
    with pytest.raises(PreregError, match="placeholder"):
        freeze(root, layout, panel, None)
    (root / "PREREG.md").write_text("H1: balanced accuracy >= 0.6\n")
    with pytest.raises(PreregError, match="calibration"):
        freeze(root, layout, panel, None)
    layout.calibration("m").write_text(json.dumps({"1": {"t1": 0.1, "t2": 0.2}}))
    with pytest.raises(PreregError, match="requires a frozen"):
        check_lock(root, layout)
    lock = freeze(root, layout, panel, None)
    assert "PREREG.md" in lock["files"] and "work/calibration/m.json" in lock["files"]
    check_lock(root, layout)
    register_test_run(layout, ["m"], None)
    with pytest.raises(PreregError, match="rerun-reason"):
        register_test_run(layout, ["m"], None)
    rec = register_test_run(layout, ["m"], "fixed a reporting bug")
    assert rec["deviations"][0]["reason"] == "fixed a reporting bug"
    layout.calibration("m").write_text(json.dumps({"1": {"t1": 0.5, "t2": 0.9}}))
    with pytest.raises(PreregError, match="changed"):
        check_lock(root, layout)
    with pytest.raises(PreregError, match="already exists"):
        freeze(root, layout, panel, None)


def test_minimap_method_names_map_to_references():
    from prp.minimap import method_reference

    assert method_reference("minimap2") == "full"
    assert method_reference("minimap2_train") == "train"
    assert method_reference("minimap2_other") is None
    assert method_reference("kmer_head") is None
