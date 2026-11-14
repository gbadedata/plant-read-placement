"""Generate a small synthetic 'phylogeny' of genomes so every stage can be tested on a laptop CPU.

Composition is hierarchical (each family has its own order-2 Markov chain, genera and species
perturb it) and homology is shared within a genus (species are mutated copies of a genus
ancestor), so alignment-based and composition-based methods both have signal at the right levels.
Toy results say nothing about real plants; they only prove the pipeline runs and is wired correctly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .seqio import decode, write_fasta

TOY_TREE = {
    # family: {genus: [species...]}
    "Alphaceae": {"Alpha": ["Alpha prima", "Alpha secunda", "Alpha tertia"], "Ambix": ["Ambix una"],
                  "Arcus": ["Arcus nova"]},
    "Betaceae": {"Beta": ["Beta prima", "Beta secunda"], "Borax": ["Borax una", "Borax duo"]},
    "Gammaceae": {"Gamma": ["Gamma prima", "Gamma secunda"], "Gyrus": ["Gyrus una"]},
    "Deltaceae": {"Delta": ["Delta prima"]},
    "Epsilaceae": {"Epsilon": ["Epsilon prima"]},
}
TOY_ROLES = {
    "Alpha prima": "panel", "Ambix una": "panel", "Beta prima": "panel", "Borax una": "panel",
    "Gamma prima": "panel", "Gyrus una": "panel",
    "Alpha secunda": "val_relative", "Borax duo": "test_relative", "Gamma secunda": "test_relative",
    "Alpha tertia": "test_relative",
    "Delta prima": "val_new", "Beta secunda": "val_relative",
    "Epsilon prima": "test_new", "Arcus nova": "test_new",
}


def _chain(rng, alpha=0.5):
    return rng.dirichlet(np.full(4, alpha), size=16)  # P(next | previous two bases)


def _perturb(chain, rng, strength):
    noisy = chain * (1 - strength) + rng.dirichlet(np.ones(4), size=16) * strength
    return noisy / noisy.sum(axis=1, keepdims=True)


def _generate(chain, n, rng):
    out = np.empty(n, dtype=np.uint8)
    out[:2] = rng.integers(0, 4, 2)
    u = rng.random(n)
    cum = np.cumsum(chain, axis=1)
    for i in range(2, n):
        state = out[i - 2] * 4 + out[i - 1]
        out[i] = min(int(np.searchsorted(cum[state], u[i])), 3)
    return out


def _mutate(seq, rate, rng):
    s = seq.copy()
    m = rng.random(len(s)) < rate
    s[m] = (s[m] + rng.integers(1, 4, m.sum())) % 4
    return s


def make_toy(outdir: str | Path, genome_len: int = 120_000, seed: int = 7) -> Path:
    outdir = Path(outdir)
    (outdir / "genomes_src").mkdir(parents=True, exist_ok=True)
    (outdir / "configs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    entries = []
    for fam, genera in TOY_TREE.items():
        fchain = _chain(rng)
        for genus, species in genera.items():
            gchain = _perturb(fchain, rng, 0.15)
            ancestor = _generate(gchain, genome_len, rng)
            for sp in species:
                schain = _perturb(gchain, rng, 0.05)
                core = _mutate(ancestor, 0.03, rng)
                own = _generate(schain, genome_len // 4, rng)
                genome = np.concatenate([core[: genome_len // 2], own, core[genome_len // 2 :]])
                half = len(genome) // 2
                recs = [("chr1", decode(genome[:half])), ("chr2", decode(genome[half:]))]
                fa = outdir / "genomes_src" / (sp.replace(" ", "_") + ".fa")
                write_fasta(recs, fa)
                entries.append({"name": sp, "genus": genus, "family": fam, "order": "Toyales",
                                "role": TOY_ROLES[sp], "pretrain": "false", "fasta": str(fa.resolve())})
    (outdir / "configs" / "panel.yaml").write_text(yaml.safe_dump({"name": "toy", "species": entries}, sort_keys=False))
    exp = {
        "seed": seed, "workdir": "work",
        "regions": {"block_size": 10_000, "min_contig": 10_000, "fractions": {"train": 0.7, "val": 0.15, "test": 0.15}},
        "simulate": {"backend": "builtin", "reads": {"train": 120, "val": 60, "test": 60}, "min_length": 1000,
                     "builtin": {"length_mean": 3000, "length_sd": 1500}},
        "windows": {"size": 512, "max_per_read": 4},
        "encoders": {"kmer": {"type": "kmer", "k": 4}, "toycnn": {"type": "toycnn", "dim": 32, "device": "cpu"}},
        "head": {"hidden": 128, "out_dim": 32, "steps": 300, "eval_every": 100, "reads_per_species": 16, "device": "cpu"},
        "finetune": {"encoder": "toycnn", "windows_per_read": 2, "reads_per_species": 4, "steps": 40, "eval_every": 20},
        "evaluation": {"specimen_sizes": [1, 5], "max_specimens_per_species": 30, "bootstrap": 200},
        "minimap2": {"threads": 2},
    }
    (outdir / "configs" / "experiment.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))
    (outdir / "PREREG.md").write_text(
        "# Toy preregistration\n\nH1: kmer_head balanced accuracy at n=5 >= 0.40.\n")
    return outdir
