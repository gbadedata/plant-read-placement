"""Window encoders: map (B, L) uint8 windows (A=0 C=1 G=2 T=3 N=4) to (B, D) float vectors.

- KmerEncoder: canonical k-mer frequencies; the alignment-free composition baseline.
- PlantCADEncoder: PlantCaduceus / PlantCAD2 via Hugging Face (needs an NVIDIA GPU and mamba-ssm).
- ToyCNNEncoder: tiny trainable CNN used to test the fine-tuning path on CPU.

All encoders are strand-invariant: k-mers are canonicalised, and PlantCaduceus averages its forward
and reverse-complement channels, so a read and its reverse complement embed identically.
"""

from __future__ import annotations

import re
from itertools import product

import numpy as np
import torch
from torch import nn


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --------------------------------------------------------------------------- k-mer baseline

def canonical_kmer_table(k: int) -> tuple[np.ndarray, int]:
    """Map every k-mer index (base-4) to a compact canonical index."""
    n = 4**k
    table = np.empty(n, dtype=np.int64)
    canon_ids: dict[int, int] = {}
    for idx, kmer in enumerate(product(range(4), repeat=k)):
        rc = 0
        for b in reversed(kmer):
            rc = rc * 4 + (3 - b)
        c = min(idx, rc)
        if c not in canon_ids:
            canon_ids[c] = len(canon_ids)
        table[idx] = canon_ids[c]
    return table, len(canon_ids)


class KmerEncoder:
    trainable = False

    def __init__(self, k: int = 5):
        self.k = k
        self.name = f"kmer{k}"
        self.table, self.dim = canonical_kmer_table(k)
        self.weights = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)

    def embed_windows(self, windows: np.ndarray) -> np.ndarray:
        w = np.asarray(windows, dtype=np.int64)
        B, L = w.shape
        k = self.k
        out = np.zeros((B, self.dim), dtype=np.float32)
        if L < k:
            return out
        # sliding k-mers via strided view
        view = np.lib.stride_tricks.sliding_window_view(w, k, axis=1)  # (B, L-k+1, k)
        valid = (view < 4).all(axis=2)
        idx = (np.minimum(view, 3) * self.weights).sum(axis=2)
        canon = self.table[idx]
        rows = np.broadcast_to(np.arange(B)[:, None], canon.shape)
        np.add.at(out, (rows[valid], canon[valid]), 1.0)
        totals = out.sum(axis=1, keepdims=True)
        np.divide(out, np.maximum(totals, 1.0), out=out)
        return out


# --------------------------------------------------------------------------- PlantCaduceus

def combine_strands(hidden: torch.Tensor, rc_equivariant: bool) -> torch.Tensor:
    """(B, T, C) hidden states -> (B, T, H), averaged over the two strands position by position.

    Caduceus RCPS models emit 2H channels. The second half is the reverse-complement stream stored
    flipped along BOTH sequence and channel axes (modeling_caduceus.py:
    `torch.cat([fwd, rc.flip(dims=[-2, -1])], -1)`), so it is already aligned to forward positions
    and only its channel order is reversed. Flip channels and average, exactly as the PlantCaduceus
    example notebook does (`reverse[..., ::-1]`). This makes the pooled embedding of a window equal
    to that of its reverse complement.
    """
    if not rc_equivariant:
        return hidden
    H = hidden.shape[-1] // 2
    fwd = hidden[..., :H]
    rc = torch.flip(hidden[..., H:], dims=[-1])
    return (fwd + rc) / 2


class PlantCADEncoder(nn.Module):
    trainable = True

    def __init__(self, model: str, device: str = "cuda", amp: str = "bf16", rc_combine: str = "auto",
                 trainable_regex: list[str] | None = None, local_files_only: bool = False,
                 components: tuple | None = None):
        """`components=(tokenizer, hf_model)` injects pre-built objects (used by the unit tests)."""
        super().__init__()
        self.model_name = model
        self.name = "plantcad"
        self.device_ = resolve_device(device)
        self.amp = amp
        if components is not None:
            tok, hf_model = components
        else:
            try:
                from transformers import AutoModelForMaskedLM, AutoTokenizer
            except ImportError as e:  # pragma: no cover - GPU environment only
                raise ImportError("pip install transformers (see docs/RUNBOOK.md)") from e
            tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=local_files_only)
            hf_model = AutoModelForMaskedLM.from_pretrained(model, trust_remote_code=True,
                                                            local_files_only=local_files_only)
        self.model = hf_model.to(self.device_)
        if amp == "bf16" and self.device_.type == "cuda" and torch.cuda.get_device_capability(self.device_) < (8, 0):
            raise RuntimeError(f"amp: bf16 needs an Ampere or newer GPU (found {torch.cuda.get_device_name(self.device_)}); "
                               "use fp16 or none")
        # Token ids for our codes A,C,G,T,N, taken from the same encode path the PlantCaduceus example uses
        # (tokenizer vocabularies differ: PlantCaduceus maps upper-case bases through its own normalisation,
        # so a direct vocabulary lookup of "A" can return the unknown-token id), plus any special tokens the
        # tokenizer adds around a sequence by default.
        def _ids(text: str, special: bool) -> list[int]:
            return list(tok.encode_plus(text, add_special_tokens=special, return_attention_mask=False,
                                        return_token_type_ids=False)["input_ids"])

        plain = _ids("ACGTN", special=False)
        if len(plain) != 5 or [_ids(b, special=False) for b in "ACGTN"] != [[i] for i in plain]:
            raise RuntimeError(f"tokenizer is not single-nucleotide: 'ACGTN' -> {plain}")
        if len(set(plain[:4])) != 4:
            raise RuntimeError(f"A, C, G and T do not map to four distinct token ids: {plain[:4]}")
        full = _ids("ACGTN", special=True)
        pos = next((i for i in range(len(full) - 4) if full[i : i + 5] == plain), None)
        if pos is None:
            raise RuntimeError(f"cannot locate nucleotide tokens in tokenizer output {full} vs {plain}")
        self.token_ids = {"A": plain[0], "C": plain[1], "G": plain[2], "T": plain[3], "N": plain[4],
                          "prefix": full[:pos], "suffix": full[pos + 5 :]}
        self.prefix = torch.tensor(full[:pos], dtype=torch.long)
        self.suffix = torch.tensor(full[pos + 5 :], dtype=torch.long)
        self.lut = torch.tensor(plain, dtype=torch.long)
        cfg = self.model.config
        if rc_combine == "auto":
            if not hasattr(cfg, "rcps"):
                print("[encoder] WARNING: model config has no 'rcps' flag; not combining strands. "
                      "Set encoders.<name>.rc_combine: true if the model is reverse-complement equivariant.")
            self.rc = bool(getattr(cfg, "rcps", False))
        else:
            self.rc = rc_combine in ("true", True)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_trainable = 0
        if trainable_regex:
            pats = [re.compile(r) for r in trainable_regex]
            for n, p in self.model.named_parameters():
                if any(pt.search(n) for pt in pats):
                    p.requires_grad_(True)
                    self.n_trainable += p.numel()
            if self.n_trainable == 0:
                names = [n for n, _ in self.model.named_parameters()][:40]
                raise RuntimeError(f"trainable_regex matched no parameters; first names: {names}")
        self.dim = None  # set on first forward

    def param_names(self) -> list[str]:
        return [n for n, _ in self.model.named_parameters()]

    def _ids(self, windows: torch.Tensor) -> torch.Tensor:
        B = windows.shape[0]
        body = self.lut.to(windows.device)[windows.long()]
        parts = []
        if len(self.prefix):
            parts.append(self.prefix.to(windows.device).expand(B, -1))
        parts.append(body)
        if len(self.suffix):
            parts.append(self.suffix.to(windows.device).expand(B, -1))
        return torch.cat(parts, dim=1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        windows = windows.to(self.device_)
        ids = self._ids(windows)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.amp)
        ctx = torch.autocast(device_type=self.device_.type, dtype=dtype) if dtype else _nullctx()
        with ctx:
            out = self.model(input_ids=ids, output_hidden_states=True)
        h = out.hidden_states[-1].float()
        a, b = len(self.prefix), h.shape[1] - len(self.suffix)
        h = combine_strands(h, self.rc)[:, a:b]
        emb = h.mean(dim=1)
        self.dim = emb.shape[-1]
        return emb

    @torch.no_grad()
    def embed_windows(self, windows: np.ndarray) -> np.ndarray:
        was = self.training
        self.eval()
        out = self.forward(torch.from_numpy(np.asarray(windows))).cpu().numpy()
        self.train(was)
        return out


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------- toy CNN (tests only)

class ToyCNNEncoder(nn.Module):
    trainable = True

    def __init__(self, dim: int = 64, device: str = "cpu", seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.name = "toycnn"
        self.dim = dim
        self.device_ = resolve_device(device)
        self.emb = nn.Embedding(5, 16)
        self.net = nn.Sequential(
            nn.Conv1d(16, 64, 9, padding=4), nn.GELU(),
            nn.Conv1d(64, dim, 9, padding=4), nn.GELU(),
        )
        self.n_trainable = sum(p.numel() for p in self.parameters())
        self.to(self.device_)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        x = self.emb(windows.long().to(self.device_)).transpose(1, 2)
        return self.net(x).mean(dim=2)

    @torch.no_grad()
    def embed_windows(self, windows: np.ndarray) -> np.ndarray:
        was = self.training
        self.eval()
        out = self.forward(torch.from_numpy(np.asarray(windows))).cpu().numpy()
        self.train(was)
        return out


def make_encoder(name: str, exp, trainable_regex: list[str] | None = None, checkpoint: str | None = None):
    cfg = exp["encoders"][name]
    t = cfg["type"]
    if t == "kmer":
        return KmerEncoder(k=int(cfg.get("k", 5)))
    if t == "plantcad":
        enc = PlantCADEncoder(cfg["model"], device=cfg.get("device", "cuda"), amp=cfg.get("amp", "bf16"),
                              rc_combine=str(cfg.get("rc_combine", "auto")).lower(),
                              trainable_regex=trainable_regex,
                              local_files_only=bool(cfg.get("local_files_only", False)))
    elif t == "toycnn":
        enc = ToyCNNEncoder(dim=int(cfg.get("dim", 64)), device=cfg.get("device", "cpu"), seed=exp.seed)
    else:
        raise ValueError(f"unknown encoder type {t}")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = enc.load_state_dict(state["encoder"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys loading {checkpoint}: {unexpected[:5]}")
        print(f"[encoder] loaded fine-tuned weights from {checkpoint} ({len(missing)} keys kept from base)")
    return enc
