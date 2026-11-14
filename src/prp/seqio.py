"""Minimal FASTA/FASTQ I/O and nucleotide encoding.

Kept dependency-free so the pipeline runs anywhere. Handles gzip transparently.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import numpy as np

# A=0 C=1 G=2 T=3 N(or anything else)=4
_ENC = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _ENC[_b] = _i
    _ENC[ord(chr(_b).lower())] = _i
_DEC = np.frombuffer(b"ACGTN", dtype=np.uint8)
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def open_text(path: str | Path, mode: str = "rt") -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode)  # type: ignore[return-value]
    return open(path, mode)


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (record_id, sequence). The id is the first whitespace token of the header."""
    name: str | None = None
    chunks: list[str] = []
    with open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].split()[0] if len(line) > 1 else ""
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


def write_fasta(records: Iterator[tuple[str, str]] | list[tuple[str, str]], path: str | Path, width: int = 80) -> int:
    n = 0
    with open_text(path, "wt") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i : i + width] + "\n")
            n += 1
    return n


def read_fastq(path: str | Path) -> Iterator[tuple[str, str, str]]:
    """Yield (full_header_without_@, sequence, quality)."""
    with open_text(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                return
            seq = fh.readline().rstrip("\n\r")
            plus = fh.readline()
            qual = fh.readline().rstrip("\n\r")
            if not header.startswith("@") or not plus.startswith("+"):
                raise ValueError(f"Malformed FASTQ record near header {header!r} in {path}")
            yield header[1:].rstrip("\n\r"), seq, qual


def write_fastq(records, path: str | Path) -> int:
    n = 0
    with open_text(path, "wt") as fh:
        for header, seq, qual in records:
            fh.write(f"@{header}\n{seq}\n+\n{qual}\n")
            n += 1
    return n


def encode(seq: str) -> np.ndarray:
    """ACGT -> 0..3, anything else -> 4. Case-insensitive."""
    return _ENC[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]


def decode(arr: np.ndarray) -> str:
    return _DEC[np.asarray(arr, dtype=np.uint8)].tobytes().decode("ascii")


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def revcomp_codes(arr: np.ndarray) -> np.ndarray:
    """Reverse complement of encoded array (N stays N)."""
    out = np.where(arr < 4, 3 - arr, 4).astype(np.uint8)
    return out[..., ::-1].copy()


def fasta_lengths(path: str | Path) -> dict[str, int]:
    return {name: len(seq) for name, seq in read_fasta(path)}


__all__ = [
    "decode",
    "encode",
    "fasta_lengths",
    "open_text",
    "read_fasta",
    "read_fastq",
    "revcomp",
    "revcomp_codes",
    "write_fasta",
    "write_fastq",
]
