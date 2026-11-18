"""Single source of truth for where every artefact lives under the work directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    root: Path  # work directory (large, git-ignored)
    project: Path | None = None  # repository root (small files that get committed)

    def _d(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # genomes
    @property
    def genomes(self) -> Path:
        return self._d("genomes")

    @property
    def lock(self) -> Path:
        return self.genomes / "lock.json"

    def genome_fasta(self, sid: str) -> Path:
        return self.genomes / f"{sid}.fa.gz"

    # regions and reads
    def region_fasta(self, sid: str, split: str) -> Path:
        return self._d("regions") / f"{sid}.{split}.fa"

    def region_manifest(self) -> Path:
        return self._d("regions") / "blocks.tsv"

    def reads(self, sid: str, split: str) -> Path:
        return self._d("reads") / f"{sid}.{split}.fastq.gz"

    # windows store
    def windows(self, split: str) -> Path:
        return self._d("windows") / f"{split}.npz"

    def read_table(self, split: str) -> Path:
        return self._d("windows") / f"{split}.reads.tsv"

    # features and models
    def features(self, name: str, split: str) -> Path:
        return self._d("features", name) / f"{split}.npy"

    def model_dir(self, name: str) -> Path:
        return self._d("models", name)

    # scores, calibration, results
    def scores(self, method: str, split: str, n: int) -> Path:
        return self._d("scores", method) / f"{split}.n{n}.tsv"

    def minimap_hits(self, split: str, reference: str = "full") -> Path:
        name = "minimap2" if reference == "full" else f"minimap2_{reference}"
        return self._d("baselines", name) / f"{split}.hits.tsv"

    def calibration(self, method: str) -> Path:
        return self._d("calibration") / f"{method}.json"

    def results(self, split: str) -> Path:
        return self._d("results", split)

    @property
    def prereg_lock(self) -> Path:
        return (self.project or self.root) / "prereg.lock.json"

    @property
    def test_marker(self) -> Path:
        return (self.project or self.root) / "TEST_RUN.json"
