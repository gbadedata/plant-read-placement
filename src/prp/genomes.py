"""Resolve and download reference genomes with the NCBI Datasets CLI, recording provenance.

Every genome that enters the benchmark is pinned in work/genomes/lock.json with its accession,
organism name as NCBI reports it, assembly level, length and the SHA-256 of the FASTA we used.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .config import Panel, Species
from .paths import Layout

LEVEL_RANK = {"complete genome": 4, "chromosome": 3, "scaffold": 2, "contig": 1}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _to_int(x: Any) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def choose_assembly(records: list[dict], species_name: str) -> dict | None:
    """Pick the best assembly for a species from `datasets summary ... --as-json-lines` records.

    Preference: NCBI reference genome > RefSeq > assembly level > contig N50. Records whose
    organism name does not start with the requested species name (e.g. a different species
    returned by a broad taxon match) are discarded; subspecies/cultivars are kept.
    """
    want = species_name.lower().strip()
    ok = [r for r in records if str(_get(r, "organism", "organism_name", default="")).lower().startswith(want)]
    if not ok:
        return None

    def key(r: dict) -> tuple:
        is_ref = str(_get(r, "assembly_info", "refseq_category", default="")).lower() == "reference genome"
        is_refseq = "refseq" in str(r.get("source_database", "")).lower() or str(r.get("accession", "")).startswith("GCF_")
        level = LEVEL_RANK.get(str(_get(r, "assembly_info", "assembly_level", default="")).lower(), 0)
        n50 = _to_int(_get(r, "assembly_stats", "contig_n50"))
        return (is_ref, is_refseq, level, n50)

    return max(ok, key=key)


def summarize(r: dict) -> dict:
    return {
        "accession": r.get("accession"),
        "organism": _get(r, "organism", "organism_name"),
        "tax_id": _get(r, "organism", "tax_id"),
        "assembly_name": _get(r, "assembly_info", "assembly_name"),
        "assembly_level": _get(r, "assembly_info", "assembly_level"),
        "refseq_category": _get(r, "assembly_info", "refseq_category"),
        "total_length": _to_int(_get(r, "assembly_stats", "total_sequence_length")),
        "contig_n50": _to_int(_get(r, "assembly_stats", "contig_n50")),
    }


def _datasets(args: list[str], exe: str = "datasets") -> str:
    if shutil.which(exe) is None:
        raise RuntimeError(
            "NCBI Datasets CLI not found. Install with: conda install -c conda-forge ncbi-datasets-cli"
        )
    res = subprocess.run([exe, *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"datasets {' '.join(args)} failed: {res.stderr.strip()[:500]}")
    return res.stdout


def query_assemblies(species: Species, exe: str = "datasets") -> list[dict]:
    if species.accession:
        out = _datasets(["summary", "genome", "accession", species.accession, "--as-json-lines"], exe)
    else:
        out = ""
        for extra in (["--reference"], []):
            try:
                out = _datasets(["summary", "genome", "taxon", species.name, *extra, "--as-json-lines"], exe)
            except RuntimeError:
                out = ""
            if out.strip():
                break
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def load_lock(layout: Layout) -> dict:
    return json.loads(layout.lock.read_text()) if layout.lock.exists() else {}


def save_lock(layout: Layout, lock: dict) -> None:
    layout.lock.write_text(json.dumps(lock, indent=2, sort_keys=True))


def resolve(panel: Panel, layout: Layout, exe: str = "datasets") -> dict:
    """Choose an assembly for each species lacking a local FASTA; write choices to the lock file."""
    lock = load_lock(layout)
    for s in panel.species:
        if s.fasta:
            lock.setdefault(s.id, {}).update({"source": "local", "fasta": s.fasta, "species": s.name})
            continue
        if s.id in lock and lock[s.id].get("accession") and not s.accession:
            continue
        recs = query_assemblies(s, exe)
        best = choose_assembly(recs, s.name) if not s.accession else (recs[0] if recs else None)
        if best is None:
            lock[s.id] = {"species": s.name, "error": "no assembly found"}
            print(f"[resolve] {s.name}: NO ASSEMBLY FOUND - pick an alternative species or pin an accession")
            continue
        info = summarize(best)
        info.update({"species": s.name, "source": "ncbi"})
        lock[s.id] = info
        print(f"[resolve] {s.name}: {info['accession']} ({info['assembly_level']}, "
              f"{info['total_length'] / 1e6:.0f} Mb, contig N50 {info['contig_n50']:,})")
    save_lock(layout, lock)
    return lock


def _gzip_copy(src: Path, dst: Path) -> None:
    opener = gzip.open if src.suffix == ".gz" else open
    with opener(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=4) as fo:
        shutil.copyfileobj(fi, fo, length=1 << 22)


def fetch(panel: Panel, layout: Layout, exe: str = "datasets") -> dict:
    """Download (or copy local) genomes to work/genomes/<id>.fa.gz and record SHA-256."""
    lock = load_lock(layout)
    for s in panel.species:
        entry = lock.get(s.id)
        if entry is None:
            raise RuntimeError(f"{s.name} not resolved; run `prp resolve` first")
        dst = layout.genome_fasta(s.id)
        if dst.exists() and entry.get("sha256") == sha256_file(dst):
            continue
        if entry.get("source") == "local":
            _gzip_copy(Path(entry["fasta"]), dst)
        elif entry.get("accession"):
            with tempfile.TemporaryDirectory() as td:
                z = Path(td) / "pkg.zip"
                _datasets(["download", "genome", "accession", entry["accession"], "--include", "genome",
                           "--filename", str(z)], exe)
                with zipfile.ZipFile(z) as zf:
                    fnas = [n for n in zf.namelist() if n.endswith((".fna", ".fa", ".fasta"))]
                    if len(fnas) != 1:
                        raise RuntimeError(f"{s.name}: expected one FASTA in package, found {fnas}")
                    zf.extract(fnas[0], td)
                    _gzip_copy(Path(td) / fnas[0], dst)
        else:
            raise RuntimeError(f"{s.name}: lock entry has no accession ({entry})")
        entry["sha256"] = sha256_file(dst)
        entry["path"] = str(dst)
        save_lock(layout, lock)
        print(f"[fetch] {s.name}: {dst.name} sha256={entry['sha256'][:12]}")
    return lock
