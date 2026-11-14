"""Simulate noisy, low-coverage Oxford Nanopore reads from each species' region FASTA.

Primary backend is Badread (Wick 2019), the standard long-read simulator. A small built-in simulator
is provided for tests and as a fallback; results produced with it must say so.

Every read header keeps the simulator's description so read identity can be parsed later, e.g.
  @Solanum_lycopersicum.test.000012 chr1:0-1000000,+strand,1234-9876 length=8123 error-free_length=8642 read_identity=91.4%
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import Experiment, Panel
from .paths import Layout
from .seqio import decode, encode, read_fasta, read_fastq, revcomp, write_fastq

IDENTITY_RE = re.compile(r"read_identity=([\d.]+)%")


def derive_seed(*parts) -> int:
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=4).digest()
    return int.from_bytes(h, "big") % (2**31 - 1)


def parse_identity(desc: str) -> float:
    m = IDENTITY_RE.search(desc)
    return float(m.group(1)) / 100.0 if m else float("nan")


def is_artifact(desc: str) -> bool:
    """Badread junk/random reads come from no genome and must not be used as training labels."""
    return ("junk_seq" in desc) or ("random_seq" in desc)


# --------------------------------------------------------------------------- built-in simulator

def _beta_params(mean: float, sd: float) -> tuple[float, float]:
    var = sd * sd
    common = mean * (1 - mean) / var - 1
    return max(mean * common, 1e-3), max((1 - mean) * common, 1e-3)


def simulate_builtin(records: list[tuple[str, str]], n_reads: int, cfg: dict, min_length: int,
                     rng: np.random.Generator) -> list[tuple[str, str, str]]:
    """Fragment + ONT-like errors: substitutions, insertions, homopolymer-biased deletions."""
    recs = [(n, s) for n, s in records if len(s) >= min_length]
    if not recs:
        raise ValueError("no region record is long enough to simulate reads")
    lengths = np.array([len(s) for _, s in recs], dtype=float)
    probs = lengths / lengths.sum()
    a, b = _beta_params(cfg["identity_mean"], cfg["identity_sd"])
    k = (cfg["length_mean"] / cfg["length_sd"]) ** 2
    theta = cfg["length_sd"] ** 2 / cfg["length_mean"]
    out = []
    while len(out) < n_reads:
        ri = rng.choice(len(recs), p=probs)
        name, seq = recs[ri]
        L = int(rng.gamma(k, theta))
        L = min(max(L, min_length), len(seq))
        start = int(rng.integers(0, len(seq) - L + 1))
        frag = seq[start : start + L]
        if frag.count("N") > 0.1 * L:
            continue
        strand = "+" if rng.random() < 0.5 else "-"
        if strand == "-":
            frag = revcomp(frag)
        q = float(np.clip(rng.beta(a, b), cfg["identity_min"], cfg["identity_max"]))
        err = 1.0 - q
        codes = encode(frag)
        n = len(codes)
        hp = np.zeros(n, dtype=bool)
        hp[1:] |= codes[1:] == codes[:-1]
        hp[:-1] |= codes[:-1] == codes[1:]
        p_del = np.where(hp, cfg["homopolymer_del_boost"], 1.0)
        p_del = p_del * (err * cfg["del_frac"] / p_del.mean())
        p_sub = err * cfg["sub_frac"]
        p_ins = err * cfg["ins_frac"]
        r = rng.random(n)
        del_mask = r < p_del
        sub_mask = (~del_mask) & (r < p_del + p_sub)
        base = codes.copy()
        acgt = base < 4
        shift = rng.integers(1, 4, n).astype(np.uint8)
        base[sub_mask & acgt] = (base[sub_mask & acgt] + shift[sub_mask & acgt]) % 4
        ins_mask = rng.random(n) < p_ins
        ins_bases = rng.integers(0, 4, n).astype(np.uint8)
        stacked = np.stack([base, ins_bases], axis=1).reshape(-1)
        keep = np.stack([~del_mask, ins_mask], axis=1).reshape(-1)
        read = decode(stacked[keep])
        if len(read) < min_length:
            continue
        n_err = int(del_mask.sum() + sub_mask.sum() + ins_mask.sum())
        ident = 1.0 - n_err / max(n + int(ins_mask.sum()), 1)
        phred = min(int(-10 * math.log10(max(err, 1e-4))), 40)
        desc = (f"{name},{strand}strand,{start}-{start + L} length={len(read)} "
                f"error-free_length={L} read_identity={ident * 100:.2f}% simulator=builtin")
        out.append((desc, read, chr(33 + phred) * len(read)))
    return out


# --------------------------------------------------------------------------- badread

def _badread_quantity(n_reads: int, mean_length: int, oversample: float) -> str:
    kb = max(1, int(round(n_reads * mean_length * oversample / 1000)))
    return f"{kb}K"


def simulate_badread(region_fa: Path, n_reads: int, cfg: dict, min_length: int, seed: int,
                     oversample: float) -> list[tuple[str, str, str]]:
    exe = cfg.get("executable", "badread")
    if shutil.which(exe) is None:
        raise RuntimeError(f"{exe} not on PATH; install Badread or set simulate.backend: builtin")
    qty = _badread_quantity(n_reads, int(cfg.get("mean_length", 8000)), oversample)
    cmd = [exe, "simulate", "--reference", str(region_fa), "--quantity", qty, "--seed", str(seed), *cfg["args"]]
    with tempfile.NamedTemporaryFile(suffix=".fastq", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with open(tmp_path, "w") as fh:
            res = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"badread failed ({' '.join(cmd)}): {res.stderr[-800:]}")
        reads = []
        for header, seq, qual in read_fastq(tmp_path):
            if len(seq) < min_length:
                continue
            parts = header.split(maxsplit=1)
            reads.append((parts[1] if len(parts) > 1 else "", seq, qual))
    finally:
        tmp_path.unlink(missing_ok=True)
    return reads


# --------------------------------------------------------------------------- driver

def _one(job: dict) -> tuple[str, str, int, int]:
    sid, split = job["sid"], job["split"]
    rng = np.random.default_rng(job["seed"])
    region = Path(job["region"])
    if job["backend"] == "builtin":
        recs = list(read_fasta(region))
        reads = simulate_builtin(recs, job["n_reads"], job["builtin"], job["min_length"], rng)
    elif job["backend"] == "badread":
        reads = simulate_badread(region, job["n_reads"], job["badread"], job["min_length"], job["seed"],
                                 job["oversample"])
        if len(reads) < job["n_reads"]:
            more = simulate_badread(region, job["n_reads"] - len(reads), job["badread"], job["min_length"],
                                    job["seed"] + 1, job["oversample"] * 2)
            reads += more
    else:
        raise ValueError(f"unknown backend {job['backend']}")
    available = len(reads)
    if available > job["n_reads"]:
        idx = np.sort(rng.choice(available, job["n_reads"], replace=False))
        reads = [reads[i] for i in idx]
    renamed = ((f"{sid}.{split}.{i:06d} {desc}", seq, qual) for i, (desc, seq, qual) in enumerate(reads))
    n = write_fastq(renamed, job["out"])
    return sid, split, n, available


def simulate_all(panel: Panel, exp: Experiment, layout: Layout, only: list[str] | None = None,
                 workers: int | None = None) -> list[tuple[str, str, int, int]]:
    cfg = exp["simulate"]
    jobs = []
    for s in panel.species:
        if only and s.id not in only:
            continue
        for split in s.splits:
            n_reads = int(cfg["reads"][split])
            seed = derive_seed(exp.seed, s.id, split)
            jobs.append({
                "sid": s.id, "split": split, "n_reads": n_reads, "seed": seed,
                "region": str(layout.region_fasta(s.id, split)), "out": str(layout.reads(s.id, split)),
                "backend": cfg["backend"], "builtin": cfg["builtin"], "badread": cfg["badread"],
                "min_length": int(cfg["min_length"]), "oversample": float(cfg["oversample"]),
            })
    # Badread holds a whole region FASTA in memory per worker; cap the default to stay within ~32 GB RAM
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2), 4))
    results = []
    if workers == 1:
        for j in jobs:
            results.append(_one(j))
            print(f"[simulate] {results[-1][0]} {results[-1][1]}: {results[-1][2]} reads")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, j) for j in jobs]
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
                print(f"[simulate] {r[0]} {r[1]}: {r[2]} reads (of {r[3]} simulated)")
    short = [r for r in results if r[2] < 0.9 * int(cfg["reads"][r[1]])]
    if short:
        print(f"[simulate] WARNING: fewer reads than requested for {short}")
    return results
