"""Panel and experiment configuration, with the benchmark's design rules enforced at load time."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROLES = ("panel", "val_relative", "val_new", "test_relative", "test_new")
TRUTH_OF_ROLE = {
    "panel": "known",
    "val_relative": "relative",
    "test_relative": "relative",
    "val_new": "new",
    "test_new": "new",
}
SPLIT_OF_ROLE = {"val_relative": "val", "val_new": "val", "test_relative": "test", "test_new": "test"}
CLASSES = ("known", "relative", "new")


class ConfigError(ValueError):
    pass


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


@dataclass
class Species:
    name: str
    genus: str
    family: str
    role: str
    order: str = ""
    pretrain: str = "unknown"  # "true" | "false" | "unknown": in PlantCaduceus pre-training set?
    accession: str | None = None
    fasta: str | None = None
    note: str = ""

    @property
    def id(self) -> str:
        return slug(self.name)

    @property
    def truth(self) -> str:
        return TRUTH_OF_ROLE[self.role]

    @property
    def splits(self) -> tuple[str, ...]:
        """Which read splits this species contributes to."""
        if self.role == "panel":
            return ("train", "val", "test")
        return (SPLIT_OF_ROLE[self.role],)


@dataclass
class Panel:
    name: str
    species: list[Species]
    path: Path | None = None

    def by_id(self) -> dict[str, Species]:
        return {s.id: s for s in self.species}

    def panel_species(self) -> list[Species]:
        return [s for s in self.species if s.role == "panel"]

    def validate(self, strict_pretrain: bool = False) -> list[str]:
        """Raise on design errors; return warnings.

        Rules (these are the leakage and labelling controls of the benchmark):
        - roles are valid and every species id is unique;
        - at least two panel species, and at least one species in each held-out role;
        - a *_relative species' genus must be in the panel, and the species itself must not be;
        - a *_new species' genus must NOT be in the panel;
        - validation and test species are disjoint (guaranteed by single role per species);
        - held-out (novel) species must not be in the PlantCaduceus pre-training set
          (strict mode, used before freezing the preregistration, refuses "unknown").
        """
        errors: list[str] = []
        warnings: list[str] = []
        ids = [s.id for s in self.species]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            errors.append(f"duplicate species: {sorted(dupes)}")
        for s in self.species:
            if s.role not in ROLES:
                errors.append(f"{s.name}: unknown role {s.role!r} (expected one of {ROLES})")
            if str(s.pretrain).lower() not in ("true", "false", "unknown"):
                errors.append(f"{s.name}: pretrain must be true/false/unknown")
        if errors:
            raise ConfigError("; ".join(errors))

        panel = self.panel_species()
        panel_genera = {s.genus for s in panel}
        panel_names = {s.name for s in panel}
        if len(panel) < 2:
            errors.append("need at least two panel species")
        for role in ROLES[1:]:
            if not any(s.role == role for s in self.species):
                errors.append(f"no species with role {role}")
        for s in self.species:
            if s.role.endswith("_relative"):
                if s.genus not in panel_genera:
                    errors.append(f"{s.name} is {s.role} but genus {s.genus} is not in the panel")
                if s.name in panel_names:
                    errors.append(f"{s.name} cannot be both panel and {s.role}")
            if s.role.endswith("_new") and s.genus in panel_genera:
                errors.append(f"{s.name} is {s.role} but genus {s.genus} IS in the panel")
            if s.role != "panel":
                p = str(s.pretrain).lower()
                if p == "true":
                    errors.append(f"{s.name} is a held-out species but is in the pre-training set")
                elif p == "unknown":
                    msg = f"{s.name}: pre-training membership unknown"
                    if strict_pretrain:
                        errors.append(msg + " (must be confirmed false before freezing)")
                    else:
                        warnings.append(msg)
        if errors:
            raise ConfigError("; ".join(errors))
        return warnings


def load_panel(path: str | Path) -> Panel:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    species = []
    for entry in raw["species"]:
        entry = dict(entry)
        entry["pretrain"] = str(entry.get("pretrain", "unknown")).lower()
        species.append(Species(**entry))
    panel = Panel(name=raw.get("name", path.stem), species=species, path=path)
    panel.validate()
    return panel


DEFAULT_EXPERIMENT: dict[str, Any] = {
    "seed": 20261114,
    "workdir": "work",
    "regions": {"block_size": 1_000_000, "min_contig": 50_000, "fractions": {"train": 0.8, "val": 0.1, "test": 0.1}},
    "simulate": {
        "backend": "badread",
        "reads": {"train": 3000, "val": 1000, "test": 1000},
        "min_length": 1000,
        "oversample": 1.3,
        "badread": {
            "executable": "badread",
            "args": ["--error_model", "nanopore2023", "--qscore_model", "nanopore2023",
                     "--identity", "92,98,3", "--length", "8000,6000"],
            "mean_length": 8000,
        },
        "builtin": {
            "identity_mean": 0.92, "identity_sd": 0.03, "identity_min": 0.80, "identity_max": 0.99,
            "length_mean": 8000, "length_sd": 6000,
            "sub_frac": 0.40, "ins_frac": 0.25, "del_frac": 0.35, "homopolymer_del_boost": 2.0,
        },
    },
    "windows": {"size": 512, "max_per_read": 8},
    "encoders": {
        "kmer": {"type": "kmer", "k": 5},
        "plantcad": {"type": "plantcad", "model": "kuleshov-group/PlantCaduceus_l20",
                     "batch_size": 64, "amp": "bf16", "device": "cuda"},
    },
    "head": {
        "hidden": 512, "out_dim": 128, "dropout": 0.1, "temperature": 0.1,
        "lambda_genus": 0.5, "lambda_family": 0.25,
        "species_per_batch": 0, "reads_per_species": 16,
        "steps": 3000, "lr": 1.0e-3, "weight_decay": 1.0e-4, "eval_every": 250, "device": "auto",
    },
    "finetune": {
        "encoder": "plantcad", "trainable_regex": [], "windows_per_read": 3, "monitor_windows": 4,
        "species_per_batch": 0, "reads_per_species": 4, "steps": 3000,
        "lr_backbone": 1.0e-4, "lr_head": 1.0e-3, "eval_every": 250,
    },
    "evaluation": {
        "specimen_sizes": [1, 10, 50], "max_specimens_per_species": 200,
        "bootstrap": 2000, "identity_bins": [0.0, 0.88, 0.92, 0.96, 1.0],
        "length_bins": [0, 2000, 5000, 10000, 1_000_000_000],
    },
    "minimap2": {"preset": "map-ont", "threads": 8},
}


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


@dataclass
class Experiment:
    raw: dict
    path: Path | None = None
    root: Path = field(default_factory=Path.cwd)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def workdir(self) -> Path:
        w = Path(self.raw["workdir"])
        return w if w.is_absolute() else self.root / w


def load_experiment(path: str | Path | None) -> Experiment:
    raw = copy.deepcopy(DEFAULT_EXPERIMENT)
    root = Path.cwd()
    if path is not None:
        path = Path(path)
        _deep_update(raw, yaml.safe_load(path.read_text()) or {})
        root = path.resolve().parent.parent if path.resolve().parent.name == "configs" else path.resolve().parent
    fr = raw["regions"]["fractions"]
    if abs(sum(fr.values()) - 1.0) > 1e-6 or set(fr) != {"train", "val", "test"}:
        raise ConfigError("regions.fractions must have train/val/test summing to 1")
    if raw["windows"]["size"] <= 0 or raw["windows"]["max_per_read"] <= 0:
        raise ConfigError("windows.size and windows.max_per_read must be positive")
    return Experiment(raw=raw, path=path, root=root)
