from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import COMP_ID_PLANTCAD, FakePlantCADTokenizer, FakeRCPSModel, FakeTokenizer, make_panel

from prp.config import ConfigError, Species, load_experiment
from prp.encoders import KmerEncoder, PlantCADEncoder, canonical_kmer_table, combine_strands
from prp.genomes import choose_assembly
from prp.losses import hierarchical_supcon, supcon_loss
from prp.regions import assign_by_rank
from prp.seqio import decode, encode, read_fasta, read_fastq, revcomp, revcomp_codes, write_fasta, write_fastq
from prp.simulate import derive_seed, is_artifact, parse_identity, simulate_builtin
from prp.windows import window_starts

# ------------------------------------------------------------------ seqio

def test_encode_roundtrip_and_case():
    assert decode(encode("ACGTNacgtx")) == "ACGTNACGTN"


def test_revcomp_codes_matches_string(rng):
    s = decode(rng.integers(0, 5, 300).astype(np.uint8))
    assert decode(revcomp_codes(encode(s))) == revcomp(s)


def test_fasta_fastq_roundtrip_gz(tmp_path):
    write_fasta([("a", "ACGT" * 50), ("b", "TT")], tmp_path / "x.fa.gz")
    assert list(read_fasta(tmp_path / "x.fa.gz")) == [("a", "ACGT" * 50), ("b", "TT")]
    write_fastq([("r1 desc", "ACG", "III")], tmp_path / "x.fq.gz")
    assert list(read_fastq(tmp_path / "x.fq.gz")) == [("r1 desc", "ACG", "III")]


# ------------------------------------------------------------------ config rules

def test_panel_valid(panel):
    assert panel.validate() == []


@pytest.mark.parametrize("bad, msg", [
    (Species("Prunus mume", "Prunus", "Rosaceae", "val_relative", pretrain="false"), "not in the panel"),
    (Species("Solanum melongena", "Solanum", "Solanaceae", "test_new", pretrain="false"), "IS in the panel"),
    (Species("Arabidopsis lyrata", "Solanum", "Brassicaceae", "test_relative", pretrain="true"), "pre-training"),
])
def test_panel_rules(bad, msg):
    with pytest.raises(ConfigError, match=msg):
        make_panel([bad]).validate()


def test_strict_pretrain_refuses_unknown():
    p = make_panel([Species("Capsicum annuum", "Capsicum", "Solanaceae", "test_new")])
    assert any("unknown" in w for w in p.validate())
    with pytest.raises(ConfigError, match="confirmed false"):
        p.validate(strict_pretrain=True)


def test_experiment_fraction_check(tmp_path):
    f = tmp_path / "e.yaml"
    f.write_text("regions: {fractions: {train: 0.5, val: 0.1, test: 0.1}}\n")
    with pytest.raises(ConfigError):
        load_experiment(f)


# ------------------------------------------------------------------ regions

def test_assign_by_rank_proportions_and_nonempty():
    hashes = list(np.random.default_rng(1).random(50))
    lab = assign_by_rank(hashes, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert lab.count("val") == 5 and lab.count("test") == 5 and lab.count("train") == 40
    small = assign_by_rank([0.3, 0.1, 0.9], {"train": 0.8, "val": 0.1, "test": 0.1})
    assert sorted(small) == ["test", "train", "val"]


# ------------------------------------------------------------------ simulation

def test_builtin_simulator_identity_and_length(rng):
    genome = decode(rng.integers(0, 4, 200_000).astype(np.uint8))
    cfg = {"identity_mean": 0.90, "identity_sd": 0.02, "identity_min": 0.8, "identity_max": 0.99,
           "length_mean": 4000, "length_sd": 2000, "sub_frac": 0.4, "ins_frac": 0.25, "del_frac": 0.35,
           "homopolymer_del_boost": 2.0}
    reads = simulate_builtin([("c1", genome)], 200, cfg, 1000, rng)
    idents = np.array([parse_identity(d) for d, _, _ in reads])
    assert len(reads) == 200 and all(len(s) >= 1000 for _, s, _ in reads)
    assert abs(idents.mean() - 0.90) < 0.02


def test_badread_header_parsing():
    h = "chr1:0-1000000,+strand,1234-9876 length=8123 error-free_length=8642 read_identity=91.43%"
    assert parse_identity(h) == pytest.approx(0.9143)
    assert not is_artifact(h)
    assert is_artifact("junk_seq length=1000 error-free_length=1000 read_identity=nan%")
    assert np.isnan(parse_identity("random_seq length=500"))


def test_derive_seed_differs():
    assert derive_seed(1, "Solanum_lycopersicum", "train") != derive_seed(1, "Solanum_tuberosum", "train")


def test_window_starts():
    assert list(window_starts(1500, 512, 8)) == [0, 988]
    s = window_starts(10_000, 512, 8)
    assert len(s) == 8 and s[0] == 0 and s[-1] == 10_000 - 512
    assert len(window_starts(400, 512, 8)) == 0
    assert list(window_starts(600, 512, 1)) == [44]


# ------------------------------------------------------------------ encoders

def test_kmer_table_size():
    _, n5 = canonical_kmer_table(5)
    _, n4 = canonical_kmer_table(4)
    assert n5 == 512 and n4 == 136  # odd k: 4^k/2; even k: (4^k + 4^(k/2))/2


def test_kmer_strand_invariant_and_normalised(rng):
    enc = KmerEncoder(5)
    w = rng.integers(0, 4, (3, 512)).astype(np.uint8)
    w[0, 100:110] = 4  # some N
    rc = np.stack([revcomp_codes(x) for x in w])
    a, b = enc.embed_windows(w), enc.embed_windows(rc)
    assert np.allclose(a, b)
    assert np.allclose(a.sum(1), 1.0)


def test_combine_strands_flips_channels_not_positions():
    f, g = torch.randn(2, 5, 3), torch.randn(2, 5, 3)
    h = torch.cat([f, torch.flip(g, dims=[-1])], dim=-1)  # RC half stored channel-reversed, position-aligned
    assert torch.allclose(combine_strands(h, True), (f + g) / 2)
    assert combine_strands(h, False).shape[-1] == 6


def test_plantcad_wrapper_tokens_and_strand_invariance(rng):
    enc = PlantCADEncoder("fake", device="cpu", amp="none", components=(FakeTokenizer(), FakeRCPSModel(6)))
    assert enc.prefix.tolist() == [] and enc.suffix.tolist() == [1] and enc.rc
    w = rng.integers(0, 4, (4, 512)).astype(np.uint8)
    ids = enc._ids(torch.from_numpy(w))
    assert ids.shape == (4, 513) and int(ids[0, -1]) == 1 and set(ids[:, :-1].unique().tolist()) <= {7, 8, 9, 10}
    a = enc.embed_windows(w)
    b = enc.embed_windows(np.stack([revcomp_codes(x) for x in w]))
    assert a.shape == (4, 6) and np.allclose(a, b, atol=1e-5)


def test_plantcad_tokenizer_behaviour_seen_on_kaggle(rng):
    tok = FakePlantCADTokenizer()
    assert tok.convert_tokens_to_ids(list("ACGTN")) == [2, 2, 2, 2, 2]  # the lookup that broke the first version
    enc = PlantCADEncoder("fake", device="cpu", amp="none",
                          components=(tok, FakeRCPSModel(6, comp=COMP_ID_PLANTCAD)))
    assert enc.lut.tolist() == [3, 4, 5, 6, 2] and enc.prefix.tolist() == [] and enc.suffix.tolist() == []
    w = rng.integers(0, 4, (4, 512)).astype(np.uint8)
    ids = enc._ids(torch.from_numpy(w))
    assert ids.shape == (4, 512) and set(ids.unique().tolist()) <= {3, 4, 5, 6}
    a = enc.embed_windows(w)
    b = enc.embed_windows(np.stack([revcomp_codes(x) for x in w]))
    assert np.allclose(a, b, atol=1e-5)


def test_tokenizer_that_is_not_single_nucleotide_is_refused():
    class KmerTok:
        def encode_plus(self, seq, add_special_tokens=True, **kw):
            return {"input_ids": [7] * max(1, len(seq) // 2)}

    with pytest.raises(RuntimeError, match="not single-nucleotide"):
        PlantCADEncoder("fake", device="cpu", amp="none", components=(KmerTok(), FakeRCPSModel(6)))


def test_plantcad_trainable_regex():
    enc = PlantCADEncoder("fake", device="cpu", amp="none", components=(FakeTokenizer(), FakeRCPSModel(6)),
                          trainable_regex=[r"^backbone\."])
    assert enc.n_trainable == 6 * 6 + 6
    with pytest.raises(RuntimeError, match="matched no parameters"):
        PlantCADEncoder("fake", device="cpu", amp="none", components=(FakeTokenizer(), FakeRCPSModel(6)),
                        trainable_regex=["does_not_exist"])


# ------------------------------------------------------------------ losses

def test_supcon_prefers_clustered_embeddings():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    clustered = torch.tensor([[1.0, 0], [0.99, 0.1], [0, 1.0], [0.1, 0.99], [-1.0, 0], [-0.99, -0.1]])
    shuffled = clustered[[0, 2, 4, 1, 3, 5]]
    assert supcon_loss(clustered, labels, 0.1) < supcon_loss(shuffled, labels, 0.1)


def test_supcon_no_positives_is_zero_and_differentiable():
    z = torch.randn(4, 8, requires_grad=True)
    loss = supcon_loss(z, torch.arange(4), 0.1)
    loss.backward()
    assert loss.item() == 0.0


def test_hierarchical_supcon_runs():
    z = torch.randn(8, 4, requires_grad=True)
    s = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    g = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    f = torch.zeros(8, dtype=torch.long)
    loss, parts = hierarchical_supcon(z, s, g, f, 0.1, 0.5, 0.25)
    loss.backward()
    assert set(parts) == {"species", "genus", "family"} and z.grad is not None


# ------------------------------------------------------------------ genome choice

def test_choose_assembly_prefers_reference_then_refseq():
    recs = [
        {"accession": "GCA_1", "organism": {"organism_name": "Solanum tuberosum"},
         "assembly_info": {"assembly_level": "Chromosome"}, "assembly_stats": {"contig_n50": 9_000_000}},
        {"accession": "GCF_2", "organism": {"organism_name": "Solanum tuberosum"},
         "assembly_info": {"assembly_level": "Scaffold", "refseq_category": "reference genome"},
         "assembly_stats": {"contig_n50": 100}},
        {"accession": "GCF_3", "organism": {"organism_name": "Solanum tuberosoides"},
         "assembly_info": {"assembly_level": "Chromosome", "refseq_category": "reference genome"}},
    ]
    assert choose_assembly(recs, "Solanum tuberosum")["accession"] == "GCF_2"
    assert choose_assembly(recs[:1], "Solanum tuberosum")["accession"] == "GCA_1"
    assert choose_assembly(recs, "Coffea arabica") is None



@pytest.mark.skipif(__import__("shutil").which("badread") is None, reason="Badread not installed")
def test_badread_wrapper_real(tmp_path, rng):
    from prp.simulate import simulate_badread

    write_fasta([("c1", decode(rng.integers(0, 4, 60_000).astype(np.uint8)))], tmp_path / "r.fa")
    cfg = {"executable": "badread", "mean_length": 3000,
           "args": ["--error_model", "nanopore2023", "--qscore_model", "nanopore2023",
                    "--identity", "92,98,3", "--length", "3000,1500"]}
    reads = simulate_badread(tmp_path / "r.fa", 30, cfg, 1000, seed=3, oversample=1.3)
    assert len(reads) >= 20
    idents = [parse_identity(d) for d, _, _ in reads if not is_artifact(d)]
    assert 0.85 < np.nanmean(idents) < 0.97


def test_gpu_check_on_fake_model(rng):
    from prp.gpucheck import layer_report, run_check

    enc = PlantCADEncoder("fake", device="cpu", amp="none", components=(FakeTokenizer(), FakeRCPSModel(6)))
    sample = rng.integers(0, 4, (16, 512)).astype(np.uint8)
    rep = run_check(enc, sample, "none", 8, 1000, [r"^backbone\."])
    assert rep["ok"] and rep["strand_invariance_min_cos"] > 0.999 and rep["layers"]["matched_tensors"] == 2
    bad = run_check(enc, sample, "none", 8, 1000, ["no_such_layer"])
    assert not bad["ok"] and "matches nothing" in bad["problems"][0]
    names = [f"caduceus.backbone.layers.{i}.mixer.in_proj.weight" for i in range(20)] + ["caduceus.backbone.norm_f.weight"]
    lr = layer_report(names, [r"layers\.(1[6-9])\.", "norm_f"])
    assert lr["n_layers"] == 20 and lr["matched_tensors"] == 5
    assert lr["suggested_patterns"] == [r"layers\.(16|17|18|19)\.", "norm_f"]


def test_finetune_shrinks_batch_on_first_step_oom(tmp_path, monkeypatch):
    """Simulate a GPU that only fits 8 windows per step: the run must shrink the batch, log it, and finish."""
    import json

    import yaml

    from prp.cli import main
    from prp.encoders import ToyCNNEncoder
    from prp.toy import make_toy

    out = make_toy(tmp_path / "toy", genome_len=40_000)
    cfg_path = out / "configs" / "experiment.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["simulate"]["reads"] = {"train": 20, "val": 10, "test": 10}
    cfg["finetune"].update({"steps": 3, "eval_every": 3, "reads_per_species": 4, "windows_per_read": 2})
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(out)
    base = ["--panel", "configs/panel.yaml", "--config", "configs/experiment.yaml"]
    for step in (["resolve"], ["fetch"], ["split"], ["simulate", "--workers", "1"], ["windows"]):
        main(step + base)
    real_forward = ToyCNNEncoder.forward

    def small_gpu(self, windows):
        if self.training and windows.shape[0] > 8:
            raise torch.OutOfMemoryError("simulated")
        return real_forward(self, windows)

    monkeypatch.setattr(ToyCNNEncoder, "forward", small_gpu)
    main(["finetune", "--encoder", "toycnn", "--name", "ft"] + base)
    log = json.loads((out / "work" / "models" / "ft" / "finetune_log.json").read_text())
    assert log["batch_reductions"], "expected the batch to be reduced"
    eff = log["effective_batch"]
    assert 6 * eff["reads_per_species"] * eff["windows_per_read"] <= 8  # six training species in the toy set
    assert (out / "work" / "models" / "ft" / "encoder.pt").exists()
    rec = log["log"][-1]
    for key in ("train_loss_mean", "untrained_loss", "skipped_steps", "encoder_grad_norm_mean",
                "encoder_weight_change_rel", "lr_backbone"):
        assert key in rec, key
    assert log["skipped_steps"] == 0  # no fp16 scaler on CPU
    assert rec["encoder_weight_change_rel"] > 0  # the encoder actually moved
    assert rec["untrained_loss"] > 0
