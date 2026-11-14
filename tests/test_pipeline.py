"""Every stage end to end on the toy data set (about a minute on a laptop CPU)."""
from __future__ import annotations

import json

import yaml

from prp.cli import main
from prp.toy import make_toy


def test_toy_pipeline_runs_and_locks(tmp_path, monkeypatch):
    out = make_toy(tmp_path / "toy", genome_len=60_000)
    cfg_path = out / "configs" / "experiment.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["simulate"]["reads"] = {"train": 60, "val": 30, "test": 30}
    cfg["head"].update({"steps": 60, "eval_every": 30})
    cfg["finetune"].update({"steps": 10, "eval_every": 5})
    cfg["evaluation"]["bootstrap"] = 50
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(out)
    main(["run-toy", "--panel", "configs/panel.yaml", "--config", "configs/experiment.yaml"])
    res = json.loads((out / "work" / "results" / "test" / "metrics.json").read_text())
    assert set(res["methods"]) == {"kmer_raw", "kmer_head", "toycnn_head", "toycnn_ft_head", "minimap2"}
    for by_n in res["methods"].values():
        for m in by_n.values():
            assert 0.0 <= m["balanced_accuracy"] <= 1.0
    # minimap2 has true homology to the panel on the toy data, so it must clear chance comfortably
    assert res["methods"]["minimap2"]["5"]["balanced_accuracy"] > 0.6
    assert (out / "prereg.lock.json").exists() and (out / "TEST_RUN.json").exists()
    assert (out / "work" / "results" / "test" / "RESULTS.md").read_text().startswith("# Plant read placement")
