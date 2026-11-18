"""Training: (1) a contrastive projection head on cached read embeddings; (2) optional adaptation of the
encoder itself (unfreezing selected PlantCaduceus layers) on noisy read windows.

Only panel species' training-region reads are used as training labels. Validation reads guide model
selection (best checkpoint) and threshold calibration; the test split is never read here.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import torch

from .embed import load_features
from .encoders import make_encoder, resolve_device
from .heads import ProjectionHead
from .losses import hierarchical_supcon
from .paths import Layout
from .placement import l2n, prototypes
from .windows import load_reads, load_windows


def _codes(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    cats = sorted(series.unique())
    m = {c: i for i, c in enumerate(cats)}
    return series.map(m).to_numpy().copy(), cats


def monitor(train_emb: np.ndarray, train_reads: pd.DataFrame, val_emb: np.ndarray, val_reads: pd.DataFrame) -> dict:
    """Read-level nearest-prototype checks on validation: species accuracy for panel species
    (held-out regions) and genus accuracy of the nearest prototype for unseen relatives."""
    protos, meta = prototypes(train_emb, train_reads)
    v = val_reads["artifact"].to_numpy() == 0
    emb = l2n(val_emb[v])
    vr = val_reads[v]
    j = np.argmax(emb @ protos.T, axis=1)
    near_sp = meta.species.to_numpy()[j]
    near_ge = meta.genus.to_numpy()[j]
    known = (vr.truth == "known").to_numpy()
    rel = (vr.truth == "relative").to_numpy()
    k_acc = float((near_sp[known] == vr.species.to_numpy()[known]).mean()) if known.any() else float("nan")
    r_acc = float((near_ge[rel] == vr.genus.to_numpy()[rel]).mean()) if rel.any() else float("nan")
    parts = [x for x in (k_acc, r_acc) if not math.isnan(x)]
    return {"known_species_acc": k_acc, "relative_genus_acc": r_acc, "score": float(np.mean(parts)) if parts else 0.0}


class BalancedSampler:
    def __init__(self, species_codes: np.ndarray, species_per_batch: int, reads_per_species: int, seed: int):
        self.by_sp = [np.flatnonzero(species_codes == c) for c in np.unique(species_codes)]
        self.P = len(self.by_sp) if species_per_batch in (0, None) else min(species_per_batch, len(self.by_sp))
        self.K = reads_per_species
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> np.ndarray:
        sps = self.rng.choice(len(self.by_sp), self.P, replace=False)
        return np.concatenate([self.rng.choice(self.by_sp[s], self.K, replace=len(self.by_sp[s]) < self.K) for s in sps])


def _cosine_lr(step: int, total: int, base: float, warmup: int = 100) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def train_head(layout: Layout, exp, features: str, out_name: str | None = None) -> dict:
    cfg = exp["head"]
    out_name = out_name or f"{features}_head"
    torch.manual_seed(exp.seed)
    device = resolve_device(cfg.get("device", "auto"))
    tr_reads = load_reads(layout, "train")
    va_reads = load_reads(layout, "val")
    X = load_features(layout, features, "train")
    Xv = load_features(layout, features, "val")
    keep = tr_reads["artifact"].to_numpy() == 0
    Xk, rk = X[keep], tr_reads[keep].reset_index(drop=True)
    sp, _ = _codes(rk.species)
    ge, _ = _codes(rk.genus)
    fa, _ = _codes(rk.family)
    head = ProjectionHead(X.shape[1], int(cfg["hidden"]), int(cfg["out_dim"]), float(cfg["dropout"]),
                          mean=torch.from_numpy(Xk.mean(0)), std=torch.from_numpy(Xk.std(0))).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    sampler = BalancedSampler(sp, int(cfg["species_per_batch"]), int(cfg["reads_per_species"]), exp.seed)
    Xt = torch.from_numpy(Xk).float().to(device)
    t_sp, t_ge, t_fa = (torch.from_numpy(a).to(device) for a in (sp, ge, fa))
    steps = int(cfg["steps"])
    log, best = [], {"score": -1.0}
    t0 = time.time()

    def embed_all(arr: np.ndarray) -> np.ndarray:
        head.eval()
        with torch.no_grad():
            out = np.concatenate([head(torch.from_numpy(arr[i : i + 8192]).float().to(device)).cpu().numpy()
                                  for i in range(0, len(arr), 8192)])
        head.train()
        return out

    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, steps, float(cfg["lr"]))
        idx = torch.from_numpy(sampler()).to(device)
        z = head(Xt[idx])
        loss, parts = hierarchical_supcon(z, t_sp[idx], t_ge[idx], t_fa[idx], float(cfg["temperature"]),
                                          float(cfg["lambda_genus"]), float(cfg["lambda_family"]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % int(cfg["eval_every"]) == 0 or step == steps - 1:
            m = monitor(embed_all(X), tr_reads, embed_all(Xv), va_reads)
            rec = {"step": step + 1, "loss": loss.item(), **parts, **m, "seconds": round(time.time() - t0, 1)}
            log.append(rec)
            print(f"[train-head] {out_name} step {step + 1}: loss {loss.item():.3f} "
                  f"val known-species {m['known_species_acc']:.3f} relative-genus {m['relative_genus_acc']:.3f}")
            if m["score"] > best["score"]:
                best = {**rec}
                torch.save({"config": head.config(), "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                            "features": features, "step": step + 1}, layout.model_dir(out_name) / "head.pt")
    summary = {"features": features, "head": out_name, "best": best, "log": log, "device": str(device),
               "config": cfg}
    (layout.model_dir(out_name) / "train_log.json").write_text(json.dumps(summary, indent=2))
    print(f"[train-head] {out_name}: best step {best.get('step')} score {best['score']:.3f}")
    return summary


# --------------------------------------------------------------------------- encoder adaptation

def finetune_encoder(layout: Layout, exp, encoder_name: str, out_name: str) -> dict:
    cfg = exp["finetune"]
    hcfg = exp["head"]
    torch.manual_seed(exp.seed)
    enc = make_encoder(encoder_name, exp, trainable_regex=list(cfg.get("trainable_regex") or []))
    if not getattr(enc, "trainable", False):
        raise ValueError(f"encoder {encoder_name} is not trainable")
    device = enc.device_
    tr_win, _ = load_windows(layout, "train")
    va_win, _ = load_windows(layout, "val")
    tr_reads = load_reads(layout, "train")
    va_reads = load_reads(layout, "val")
    keep = tr_reads["artifact"].to_numpy() == 0
    rk = tr_reads[keep].reset_index(drop=True)
    sp, _ = _codes(rk.species)
    ge, _ = _codes(rk.genus)
    fa, _ = _codes(rk.family)
    rng = np.random.default_rng(exp.seed)
    W = int(cfg["windows_per_read"])
    W_MON = int(cfg.get("monitor_windows", 4))  # windows per read when monitoring (closer to final embedding)

    def sample_windows(reads: pd.DataFrame, rows: np.ndarray, win: np.ndarray, W: int) -> torch.Tensor:
        out = []
        for r in rows:
            first, nw = int(reads.first_window.iloc[r]), int(reads.n_windows.iloc[r])
            pick = rng.choice(nw, W, replace=nw < W)
            out.append(win[first + pick])
        return torch.from_numpy(np.concatenate(out))

    with torch.no_grad():
        enc.eval()
        probe = enc(torch.from_numpy(tr_win[:128]))
        enc.train()
    dim = probe.shape[1]
    head = ProjectionHead(dim, int(hcfg["hidden"]), int(hcfg["out_dim"]), float(hcfg["dropout"]),
                          mean=probe.mean(0).cpu(), std=probe.std(0).cpu()).to(device)
    enc_params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": float(cfg["lr_backbone"]), "base": float(cfg["lr_backbone"])},
                             {"params": head.parameters(), "lr": float(cfg["lr_head"]), "base": float(cfg["lr_head"])}],
                            weight_decay=float(hcfg["weight_decay"]))
    sampler = BalancedSampler(sp, int(cfg["species_per_batch"]), int(cfg["reads_per_species"]), exp.seed)
    t_sp, t_ge, t_fa = (torch.from_numpy(a).to(device) for a in (sp, ge, fa))
    # fp16 autocast (e.g. on a T4, which has no bf16) needs loss scaling so small gradients do not underflow
    use_scaler = getattr(enc, "amp", "none") == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # small fixed monitoring subsets (full validation embedding happens later with `prp embed`)
    def subset(reads: pd.DataFrame, per_sp: int) -> np.ndarray:
        r = reads[reads.artifact == 0]
        return np.concatenate([g.read_idx.to_numpy()[:per_sp] for _, g in r.groupby("species")])

    mon_tr = subset(tr_reads, 40)
    mon_va = subset(va_reads, 40)

    def embed_reads(reads: pd.DataFrame, rows: np.ndarray, win: np.ndarray) -> np.ndarray:
        enc.eval()
        head.eval()
        outs = []
        with torch.no_grad():
            for r in rows:
                first, nw = int(reads.first_window.iloc[r]), int(reads.n_windows.iloc[r])
                e = enc(torch.from_numpy(win[first : first + min(nw, W_MON)])).mean(0, keepdim=True)
                outs.append(head(e).cpu().numpy())
        enc.train()
        head.train()
        return np.concatenate(outs)

    steps = int(cfg["steps"])
    log, best = [], {"score": -1.0}
    reductions: list[dict] = []
    t0 = time.time()
    step = 0
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, steps, g["base"])
        oom = False
        try:
            rows = sampler()
            x = sample_windows(rk, rows, tr_win, W)
            e = enc(x)
            e = e.view(len(rows), W, -1).mean(1)
            z = head(e)
            idx = torch.from_numpy(rows).to(device)
            loss, parts = hierarchical_supcon(z, t_sp[idx], t_ge[idx], t_fa[idx], float(hcfg["temperature"]),
                                              float(hcfg["lambda_genus"]), float(hcfg["lambda_family"]))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        except torch.OutOfMemoryError:
            if step > 0:
                raise
            oom = True
        if oom:
            # Only the first step may shrink the batch, and every shrink is logged: never a silent change.
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            before = {"reads_per_species": sampler.K, "windows_per_read": W}
            if sampler.K > 1:
                sampler.K //= 2
            elif W > 1:
                W //= 2
            else:
                raise RuntimeError("out of GPU memory even at one window per read; lower species_per_batch")
            change = {"from": before, "to": {"reads_per_species": sampler.K, "windows_per_read": W}}
            reductions.append(change)
            print(f"[finetune] out of GPU memory on the first step; batch reduced {change}", flush=True)
            continue
        if step == 0:
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            print(f"[finetune] first step OK: {len(rows)} reads x {W} windows per step, "
                  f"peak GPU memory {peak:.1f} GiB", flush=True)
        if (step + 1) % int(cfg["eval_every"]) == 0 or step == steps - 1:
            m = monitor(embed_reads(tr_reads, mon_tr, tr_win), tr_reads.iloc[mon_tr].reset_index(drop=True),
                        embed_reads(va_reads, mon_va, va_win), va_reads.iloc[mon_va].reset_index(drop=True))
            rec = {"step": step + 1, "loss": loss.item(), **parts, **m, "seconds": round(time.time() - t0, 1)}
            log.append(rec)
            print(f"[finetune] {out_name} step {step + 1}: loss {loss.item():.3f} "
                  f"val known-species {m['known_species_acc']:.3f} relative-genus {m['relative_genus_acc']:.3f}", flush=True)
            if m["score"] > best["score"]:
                best = {**rec}
                trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
                state = {k: v.detach().cpu() for k, v in enc.state_dict().items() if k in trainable}
                torch.save({"encoder": state, "encoder_name": encoder_name, "step": step + 1,
                            "trainable_regex": cfg.get("trainable_regex")}, layout.model_dir(out_name) / "encoder.pt")
        step += 1
    summary = {"encoder": encoder_name, "out": out_name, "n_trainable": int(enc.n_trainable), "best": best,
               "log": log, "config": cfg, "batch_reductions": reductions,
               "effective_batch": {"reads_per_species": sampler.K, "windows_per_read": W}}
    (layout.model_dir(out_name) / "finetune_log.json").write_text(json.dumps(summary, indent=2))
    print(f"[finetune] {out_name}: best step {best.get('step')} score {best['score']:.3f}; "
          f"{enc.n_trainable:,} trainable encoder parameters")
    return summary
