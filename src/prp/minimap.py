"""minimap2 baseline (via mappy): align every query read to each panel genome and keep the best hit.

Two reference sets, reported as separate methods:

- `minimap2` (reference="full"): the *full* genome of every panel species, validation and test regions
  included. Generous to the baseline, as a real reference database would contain the whole genome, but
  it means a known species' reads align to the exact sequence they were simulated from.
- `minimap2_train` (reference="train"): only the training blocks of each panel genome, the same
  sequence the learned methods were trained on. Reads from validation and test blocks then have to
  align to other parts of their own genome (repeats, gene families) or not at all. Added on 18 Nov 2026,
  before freezing, as a reference row with no hypothesis attached (PREREG.md, development log).

Read similarity = matched bases in the best alignment / read length (0 if unmapped), which folds
alignment identity and query coverage into one number. Specimen distance d1 = 1 - mean similarity.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .config import Panel
from .paths import Layout
from .placement import build_specimens
from .seqio import read_fastq
from .windows import load_reads


def _read_sequences(panel: Panel, layout: Layout, split: str) -> dict[str, str]:
    seqs = {}
    for s in panel.species:
        if split in s.splits:
            for header, seq, _ in read_fastq(layout.reads(s.id, split)):
                seqs[header.split(" ", 1)[0]] = seq
    return seqs


REFERENCES = ("full", "train")


def reference_fasta(layout: Layout, sid: str, reference: str):
    if reference == "full":
        return layout.genome_fasta(sid)
    if reference == "train":
        path = layout.region_fasta(sid, "train")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: it is written by `prp split`; re-run that step")
        return path
    raise ValueError(f"reference must be one of {REFERENCES} (got {reference})")


def method_reference(method: str) -> str | None:
    """'minimap2' -> 'full', 'minimap2_train' -> 'train', anything else -> None."""
    if method == "minimap2":
        return "full"
    if method.startswith("minimap2_") and method[len("minimap2_"):] in REFERENCES:
        return method[len("minimap2_"):]
    return None


def map_split(panel: Panel, exp, layout: Layout, split: str, reference: str = "full") -> pd.DataFrame:
    import mappy

    cfg = exp["minimap2"]
    threads = int(cfg.get("threads", 8))
    reads = load_reads(layout, split)
    seqs = _read_sequences(panel, layout, split)
    order = reads.read_id.tolist()
    n = len(order)
    best_sp = np.array([""] * n, dtype=object)
    best_mlen = np.zeros(n, dtype=np.int64)
    best_blen = np.zeros(n, dtype=np.int64)
    best_qcov = np.zeros(n, dtype=np.float64)
    lengths = reads.length.to_numpy()
    chunks = np.array_split(np.arange(n), max(1, threads * 4))
    for s in panel.panel_species():
        t0 = time.time()
        fasta = reference_fasta(layout, s.id, reference)
        aligner = mappy.Aligner(str(fasta), preset=cfg.get("preset", "map-ont"), n_threads=threads)
        if not aligner:
            raise RuntimeError(f"minimap2 failed to build an index for {s.name}")

        def work(ix: np.ndarray, aligner=aligner):
            buf = mappy.ThreadBuffer()
            out = []
            for i in ix:
                top = None
                for h in aligner.map(seqs[order[i]], buf=buf):
                    if h.is_primary and (top is None or h.mlen > top.mlen):
                        top = h
                if top is not None:
                    out.append((i, top.mlen, top.blen, (top.q_en - top.q_st) / max(lengths[i], 1)))
            return out

        with ThreadPoolExecutor(max_workers=threads) as ex:
            for res in ex.map(work, chunks):
                for i, mlen, blen, qcov in res:
                    if mlen > best_mlen[i]:
                        best_mlen[i], best_blen[i], best_qcov[i], best_sp[i] = mlen, blen, qcov, s.id
        print(f"[minimap2:{reference}] {split}: mapped against {s.name} in {time.time() - t0:.0f} s", flush=True)
        del aligner
    sim = best_mlen / np.maximum(lengths, 1)
    hits = pd.DataFrame({
        "read_idx": reads.read_idx, "read_id": reads.read_id, "best_species": best_sp,
        "mapped": (best_mlen > 0).astype(int), "mlen": best_mlen, "blen": best_blen,
        "identity": np.where(best_blen > 0, best_mlen / np.maximum(best_blen, 1), 0.0),
        "query_coverage": best_qcov, "similarity": np.clip(sim, 0, 1),
    })
    hits.to_csv(layout.minimap_hits(split, reference), sep="\t", index=False)
    print(f"[minimap2:{reference}] {split}: {hits.mapped.mean():.1%} of reads mapped", flush=True)
    return hits


def score_minimap(hits: pd.DataFrame, reads: pd.DataFrame, panel: Panel, n: int, max_per_species: int,
                  seed: int) -> pd.DataFrame:
    meta = {s.id: (s.genus, s.family) for s in panel.panel_species()}
    h = hits.set_index("read_idx")
    r = reads.set_index("read_idx", drop=False)
    rows = []
    for sid, ix in build_specimens(reads, n, max_per_species, seed):
        sub = h.loc[ix]
        first = r.loc[ix[0]]
        d1 = 1.0 - float(sub.similarity.mean())
        mapped = sub[sub.mapped == 1]
        if len(mapped):
            votes = mapped.groupby("best_species").mlen.sum().sort_values(ascending=False)
            near = votes.index[0]
            g, f = meta[near]
        else:
            near, g, f = "", "", ""
        rows.append({
            "specimen": sid, "n": n, "species": first.species, "genus": first.genus, "family": first.family,
            "truth": first.truth, "d1": d1, "d2": float("nan"), "nearest_species": near, "nearest_genus": g,
            "nearest_family": f,
            "mean_identity": float(np.nanmean(r.loc[ix, "identity"])) if r.loc[ix, "identity"].notna().any() else float("nan"),
            "mean_length": float(r.loc[ix, "length"].mean()), "artifact_frac": float(r.loc[ix, "artifact"].mean()),
            "mapped_frac": float(sub.mapped.mean()),
        })
    return pd.DataFrame(rows)
