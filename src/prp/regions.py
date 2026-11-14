"""Split each genome into disjoint train / val / test regions.

Leakage controls applied here:
- Panel species are cut into fixed-size blocks and each block is assigned to exactly one split by a
  seeded hash, so no test read can overlap a sequence the model or the prototypes were built from.
- Held-out species contribute only to their own split (val or test) and never to training.
- Sequences are upper-cased: assemblies are soft-masked (lower-case repeats) and real reads carry no
  such annotation, so keeping case would leak repeat labels into the reads.
"""

from __future__ import annotations

import csv
import hashlib

from .config import Experiment, Panel
from .paths import Layout
from .seqio import read_fasta, write_fasta

SPLITS = ("train", "val", "test")


def block_hash(seed: int, sid: str, contig: str, block: int) -> float:
    h = hashlib.blake2b(f"{seed}|{sid}|{contig}|{block}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


def assign_by_rank(hashes: list[float], fractions: dict[str, float]) -> list[str]:
    """Rank blocks by a seeded hash and cut the ranking by the split fractions.

    Unlike thresholding each hash independently, this guarantees every split gets at least one
    block (given three or more blocks) and gives exact proportions, while staying deterministic.
    """
    n = len(hashes)
    if n < 3:
        raise RuntimeError(f"only {n} usable blocks; need at least 3 to form train/val/test regions")
    n_val = max(1, round(n * fractions["val"]))
    n_test = max(1, round(n * fractions["test"]))
    if n - n_val - n_test < 1:
        raise RuntimeError("block fractions leave no training blocks; lower regions.block_size")
    order = sorted(range(n), key=lambda i: hashes[i])
    out = ["train"] * n
    for i in order[:n_val]:
        out[i] = "val"
    for i in order[n_val : n_val + n_test]:
        out[i] = "test"
    return out


def split_genomes(panel: Panel, exp: Experiment, layout: Layout, only: list[str] | None = None) -> list[dict]:
    cfg = exp["regions"]
    block_size = int(cfg["block_size"])
    min_contig = int(cfg["min_contig"])
    min_block = max(min_contig // 2, 1)
    fractions = cfg["fractions"]
    rows: list[dict] = []
    for s in panel.species:
        if only and s.id not in only:
            continue
        src = layout.genome_fasta(s.id)
        if not src.exists():
            raise FileNotFoundError(f"{src} missing; run `prp fetch` first")
        per_split: dict[str, list[tuple[str, str]]] = {sp: [] for sp in s.splits}
        blocks: list[tuple[str, int, str, float]] = []
        for contig, seq in read_fasta(src):
            if len(seq) < min_contig:
                continue
            seq = seq.upper()
            for start in range(0, len(seq), block_size):
                block = seq[start : start + block_size]
                if len(block) < min_block:
                    continue
                n_frac = block.count("N") / len(block)
                if n_frac > 0.5:
                    continue
                blocks.append((contig, start, block, n_frac))
        if s.role == "panel":
            hashes = [block_hash(exp.seed, s.id, c, st // block_size) for c, st, _, _ in blocks]
            labels = assign_by_rank(hashes, fractions)
        else:
            labels = [s.splits[0]] * len(blocks)
        for (contig, start, block, n_frac), split in zip(blocks, labels, strict=True):
            per_split[split].append((f"{contig}:{start}-{start + len(block)}", block))
            rows.append({"species": s.id, "contig": contig, "start": start, "end": start + len(block),
                         "split": split, "n_frac": round(n_frac, 4)})
        del blocks
        for split, recs in per_split.items():
            if not recs:
                raise RuntimeError(f"{s.name}: no blocks assigned to {split}; lower regions.block_size")
            write_fasta(recs, layout.region_fasta(s.id, split))
        sizes = {sp: sum(len(r[1]) for r in recs) for sp, recs in per_split.items()}
        print(f"[split] {s.name}: " + ", ".join(f"{sp}={v / 1e6:.1f} Mb" for sp, v in sizes.items()))
    manifest = layout.region_manifest()
    existing = []
    if only and manifest.exists():
        with open(manifest) as fh:
            existing = [r for r in csv.DictReader(fh, delimiter="\t") if r["species"] not in set(only)]
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["species", "contig", "start", "end", "split", "n_frac"], delimiter="\t")
        w.writeheader()
        w.writerows(existing + rows)
    return rows
