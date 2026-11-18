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
    main(["explore", "--methods", "kmer_head", "toycnn_head", "minimap2", "--panel", "configs/panel.yaml",
          "--config", "configs/experiment.yaml"])
    explore = (out / "work" / "results" / "val" / "EXPLORE_distances.md").read_text()
    assert "Not a preregistered analysis" in explore and "toycnn_head" in explore

    # Spreading splits over two "GPUs" (forced; child processes on CPU here) must give identical features.
    import numpy as np

    monkeypatch.setenv("PRP_EMBED_GPUS", "2")
    main(["embed", "--encoder", "toycnn", "--name", "toycnn_two", "--panel", "configs/panel.yaml",
          "--config", "configs/experiment.yaml"])
    for split in ("train", "val", "test"):
        a = np.load(out / "work" / "features" / "toycnn" / f"{split}.npy")
        b = np.load(out / "work" / "features" / "toycnn_two" / f"{split}.npy")
        assert np.allclose(a, b, atol=1e-6)
