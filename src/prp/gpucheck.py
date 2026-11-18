"""Five-minute pre-flight for a GPU session: does the model load, agree across precisions, stay
strand-invariant, match the fine-tuning layer patterns, and how long will the full run take?

Fails (non-zero exit through the CLI) if the embedding is not strand-invariant or if the configured
mixed precision disagrees with full precision, so a long run never starts on a broken setup.
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import torch

from .seqio import revcomp_codes


def _cos_rows(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return float((a * b).sum(1).min())


def layer_report(param_names: list[str], patterns: list[str]) -> dict:
    ids = sorted({int(m.group(1)) for n in param_names if (m := re.search(r"layers\.(\d+)\.", n))})
    matched = [n for n in param_names if any(re.search(p, n) for p in patterns)]
    top = ids[-4:] if ids else []
    suggestion = None
    if top:
        suggestion = [r"layers\.(" + "|".join(str(i) for i in top) + r")\.", "norm_f"]
    return {"n_layers": len(ids), "layer_ids": [ids[0], ids[-1]] if ids else [], "configured_patterns": patterns,
            "matched_tensors": len(matched), "matched_example": matched[:3], "suggested_patterns": suggestion}


def run_check(enc, sample: np.ndarray, configured_amp: str, batch_size: int, total_windows: int,
              patterns: list[str], min_cos: float = 0.995) -> dict:
    out: dict = {"cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        # torch reports bf16 as "supported" on pre-Ampere GPUs because it can emulate it, but Mamba's Triton
        # kernels need native bf16 (compute capability 8.0+); on a T4 (7.5) they fail to compile.
        out.update(gpu=torch.cuda.get_device_name(0), capability=list(cap), bf16_supported=cap >= (8, 0))
    names = enc.param_names() if hasattr(enc, "param_names") else [n for n, _ in enc.named_parameters()]
    out["layers"] = layer_report(names, patterns)
    out["token_ids"] = getattr(enc, "token_ids", None)
    out["rc_combine"] = getattr(enc, "rc", None)
    enc.amp = "none"
    e32 = enc.embed_windows(sample)
    erc = enc.embed_windows(np.stack([revcomp_codes(w) for w in sample]))
    out["strand_invariance_min_cos"] = _cos_rows(e32, erc)
    out["embedding_dim"] = int(e32.shape[1])
    out["precision_min_cos"] = {}
    if torch.cuda.is_available():
        for amp in ("fp16", "bf16"):
            if amp == "bf16" and not out.get("bf16_supported"):
                continue
            enc.amp = amp
            try:
                out["precision_min_cos"][amp] = _cos_rows(e32, enc.embed_windows(sample))
            except Exception as e:  # record, do not crash: the configured precision is checked below
                out["precision_min_cos"][amp] = f"failed: {type(e).__name__}: {str(e).splitlines()[0][:160]}"
    enc.amp = configured_amp
    reps, t0, n = 0, time.time(), 0
    while time.time() - t0 < 20 and reps < 50:
        enc.embed_windows(sample[:batch_size])
        n += min(batch_size, len(sample))
        reps += 1
    rate = n / max(time.time() - t0, 1e-9)
    out.update(configured_amp=configured_amp, windows_per_second=round(rate, 1), total_windows=int(total_windows),
               estimated_embed_minutes=round(total_windows / max(rate, 1e-9) / 60, 1))
    problems = []
    if out["strand_invariance_min_cos"] < 0.999:
        problems.append(f"embedding is not strand-invariant (min cosine {out['strand_invariance_min_cos']:.4f})")
    got = out["precision_min_cos"].get(configured_amp)
    if isinstance(got, str):
        problems.append(f"configured precision {configured_amp} {got}; use --amp none or set encoders.<name>.amp: none")
    elif got is not None and got < min_cos:
        problems.append(f"{configured_amp} disagrees with fp32 (min cosine {out['precision_min_cos'][configured_amp]:.4f}); "
                        "use --amp none or set encoders.<name>.amp: none")
    if patterns and out["layers"]["matched_tensors"] == 0:
        problems.append(f"finetune.trainable_regex matches nothing; try {out['layers']['suggested_patterns']}")
    out["problems"] = problems
    out["ok"] = not problems
    return out


def save(report: dict, path) -> None:
    path.write_text(json.dumps(report, indent=2))
