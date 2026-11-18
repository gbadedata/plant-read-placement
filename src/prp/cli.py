"""Command-line interface. Run `prp <command> -h` for options.

Typical order (see docs/RUNBOOK.md):
  resolve -> fetch -> split -> simulate -> windows -> embed -> [finetune -> embed] -> train-head
  -> minimap -> dev (score + calibrate + evaluate on validation) -> fill PREREG.md -> freeze
  -> evaluate --split test -> report --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import load_experiment, load_panel


def _ctx(args):
    from .paths import Layout

    panel = load_panel(args.panel)
    exp = load_experiment(args.config)
    layout = Layout(exp.workdir, exp.root)
    return panel, exp, layout


# --------------------------------------------------------------------------- method plumbing

def method_transform(layout, method: str):
    from .embed import load_features
    from .placement import HeadTransform, RawTransform, load_head

    if method.endswith("_raw"):
        feat = method[: -len("_raw")]
        return feat, RawTransform(load_features(layout, feat, "train"))
    if method.endswith("_head"):
        feat = method[: -len("_head")]
        return feat, HeadTransform(load_head(layout, method))
    raise ValueError(f"method must be minimap2, minimap2_train, <features>_raw or <features>_head (got {method})")


def compute_scores(panel, exp, layout, method: str, split: str) -> dict[int, pd.DataFrame]:
    from .embed import load_features
    from .minimap import map_split, method_reference, score_minimap
    from .placement import prototypes, score_embeddings
    from .windows import load_reads

    ev = exp["evaluation"]
    reads = load_reads(layout, split)
    out = {}
    ref = method_reference(method)
    if ref is not None:
        p = layout.minimap_hits(split, ref)
        hits = pd.read_csv(p, sep="\t") if p.exists() else map_split(panel, exp, layout, split, ref)
        for n in ev["specimen_sizes"]:
            out[int(n)] = score_minimap(hits, reads, panel, int(n), int(ev["max_specimens_per_species"]), exp.seed)
    else:
        feat, T = method_transform(layout, method)
        tr_reads = load_reads(layout, "train")
        protos, meta = prototypes(T(load_features(layout, feat, "train")), tr_reads)
        emb = T(load_features(layout, feat, split))
        for n in ev["specimen_sizes"]:
            out[int(n)] = score_embeddings(emb, reads, protos, meta, int(n), int(ev["max_specimens_per_species"]), exp.seed)
    for n, df in out.items():
        df.to_csv(layout.scores(method, split, n), sep="\t", index=False)
    return out


def load_or_compute_scores(panel, exp, layout, method, split) -> dict[int, pd.DataFrame]:
    sizes = [int(n) for n in exp["evaluation"]["specimen_sizes"]]
    paths = {n: layout.scores(method, split, n) for n in sizes}
    if all(p.exists() for p in paths.values()):
        return {n: pd.read_csv(p, sep="\t") for n, p in paths.items()}
    return compute_scores(panel, exp, layout, method, split)


# --------------------------------------------------------------------------- commands

def cmd_toy(args):
    from .toy import make_toy

    out = make_toy(args.out, genome_len=args.genome_len)
    print(f"[toy] wrote {out}. Next: cd {out} and run `prp run-toy` or the stages one by one.")


def cmd_resolve(args):
    from .genomes import resolve

    panel, exp, layout = _ctx(args)
    resolve(panel, layout, exe=args.datasets)


def cmd_fetch(args):
    from .genomes import fetch

    panel, exp, layout = _ctx(args)
    fetch(panel, layout, exe=args.datasets)


def cmd_split(args):
    from .regions import split_genomes

    panel, exp, layout = _ctx(args)
    split_genomes(panel, exp, layout, only=args.only)


def cmd_simulate(args):
    from .simulate import simulate_all

    panel, exp, layout = _ctx(args)
    if args.backend:
        exp.raw["simulate"]["backend"] = args.backend
    simulate_all(panel, exp, layout, only=args.only, workers=args.workers)


def cmd_windows(args):
    from .windows import build_windows

    panel, exp, layout = _ctx(args)
    build_windows(panel, exp, layout)


def _gpu_groups(layout, splits: list[str], n_gpus: int) -> list[list[str]]:
    """Assign splits to GPUs, largest first, so the windows each GPU embeds are roughly balanced."""
    sizes = {s: int(np.load(layout.windows(s))["read_index"].shape[0]) for s in splits}
    groups: list[list[str]] = [[] for _ in range(n_gpus)]
    load = [0] * n_gpus
    for s in sorted(splits, key=lambda x: -sizes[x]):
        g = load.index(min(load))
        groups[g].append(s)
        load[g] += sizes[s]
    return [g for g in groups if g]


def _embed_on_gpus(args, layout) -> None:
    """Run one child `prp embed` per GPU (CUDA_VISIBLE_DEVICES), each on its own splits, in parallel."""
    import os
    import subprocess
    import threading

    n = int(os.environ.get("PRP_EMBED_GPUS", "0") or 0) or __import__("torch").cuda.device_count()
    groups = _gpu_groups(layout, args.splits, n)
    print(f"[embed] {len(groups)} GPUs in parallel: " + "; ".join(f"gpu{i}: {' '.join(g)}" for i, g in enumerate(groups)), flush=True)
    procs = []
    for i, group in enumerate(groups):
        cmd = [sys.executable, "-m", "prp.cli", "embed", "--panel", args.panel, "--config", args.config,
               "--encoder", args.encoder, "--splits", *group]
        for flag, val in (("--name", args.name), ("--checkpoint", args.checkpoint),
                          ("--batch-size", args.batch_size), ("--amp", args.amp)):
            if val:
                cmd += [flag, str(val)]
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(CUDA_VISIBLE_DEVICES=str(i), PRP_EMBED_CHILD="1")
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        def pump(proc=p, tag=f"gpu{i}"):
            for line in proc.stdout:
                print(f"[{tag}] {line}", end="", flush=True)

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        procs.append((p, t))
    codes = []
    for p, t in procs:
        codes.append(p.wait())
        t.join()
    if any(codes):
        sys.exit(f"error: embedding failed on GPU(s) {[i for i, c in enumerate(codes) if c]}")


def cmd_embed(args):
    import os

    from .embed import embed_split
    from .encoders import make_encoder

    panel, exp, layout = _ctx(args)
    etype = exp["encoders"][args.encoder]["type"]
    if etype in ("plantcad", "toycnn") and len(args.splits) > 1 and not os.environ.get("PRP_EMBED_CHILD") \
            and not args.one_process:
        n = int(os.environ.get("PRP_EMBED_GPUS", "0") or 0)
        if not n:
            import torch

            n = torch.cuda.device_count()
        if n > 1:
            _embed_on_gpus(args, layout)
            return
    if args.amp:
        exp.raw["encoders"][args.encoder]["amp"] = args.amp
    enc = make_encoder(args.encoder, exp, checkpoint=args.checkpoint)
    name = args.name or (args.encoder + ("_ft" if args.checkpoint else ""))
    bs = args.batch_size or int(exp["encoders"][args.encoder].get("batch_size", 256))
    meta = {"encoder": args.encoder, "checkpoint": args.checkpoint,
            "model": exp["encoders"][args.encoder].get("model"), "amp": exp["encoders"][args.encoder].get("amp")}
    for split in args.splits:
        embed_split(enc, layout, split, name, batch_size=bs, meta=meta)


def cmd_train_head(args):
    from .train import train_head

    panel, exp, layout = _ctx(args)
    train_head(layout, exp, args.features)


def cmd_finetune(args):
    from .train import finetune_encoder

    panel, exp, layout = _ctx(args)
    name = args.encoder or exp["finetune"]["encoder"]
    if args.amp:
        exp.raw["encoders"][name]["amp"] = args.amp
    finetune_encoder(layout, exp, name, args.name)


def cmd_minimap(args):
    from .minimap import map_split

    panel, exp, layout = _ctx(args)
    for split in args.splits:
        map_split(panel, exp, layout, split, args.reference)


def cmd_dev(args):
    """Score, calibrate and evaluate on validation for each method."""
    from .metrics import evaluate
    from .placement import calibrate, save_calibration

    panel, exp, layout = _ctx(args)
    results = {"split": "val", "methods": {}}
    for method in args.methods:
        scores = compute_scores(panel, exp, layout, method, "val")
        calib = {}
        results["methods"][method] = {}
        for n, df in scores.items():
            calib[str(n)] = calibrate(df)
            results["methods"][method][str(n)] = evaluate(df, calib[str(n)], exp, exp.seed)
            m = results["methods"][method][str(n)]
            print(f"[dev] {method} n={n}: val balanced accuracy {m['balanced_accuracy']:.3f} "
                  f"(known {m['recall_known']:.2f}, relative {m['recall_relative']:.2f}, new {m['recall_new']:.2f})")
        save_calibration(layout, method, calib)
    _write_results(layout, "val", results)


def _write_results(layout, split, results):
    from .report import write_report

    rdir = layout.results(split)
    existing = rdir / "metrics.json"
    if existing.exists() and split == "val":
        old = json.loads(existing.read_text())
        old["methods"].update(results["methods"])
        results = old
    existing.write_text(json.dumps(results, indent=2, default=float))
    out = write_report(layout, split, title="Plant read placement")
    print(f"[results] {split}: {out}")


def cmd_freeze(args):
    from .prereg import freeze

    panel, exp, layout = _ctx(args)
    lock = freeze(exp.root, layout, panel, Path(args.config) if args.config else None)
    print(f"[freeze] preregistration frozen at {lock['frozen_at']} ({len(lock['files'])} files hashed)")
    for f in lock["files"]:
        print(f"  {f}")


def cmd_evaluate(args):
    from .metrics import evaluate
    from .placement import load_calibration
    from .prereg import check_lock, register_test_run

    panel, exp, layout = _ctx(args)
    if args.split == "test":
        lock = check_lock(exp.root, layout)
        frozen = set(lock["files"])
        for m in args.methods:
            cal = layout.calibration(m).resolve()
            rel = str(cal.relative_to(exp.root.resolve())) if cal.is_relative_to(exp.root.resolve()) else str(cal)
            if rel not in frozen:
                sys.exit(f"{m}: calibration {rel} is not part of the frozen preregistration")
        register_test_run(layout, args.methods, args.rerun_reason)
        for m in args.methods:
            compute_scores(panel, exp, layout, m, "test")
    results = {"split": args.split, "methods": {}}
    for m in args.methods:
        calib = load_calibration(layout, m)
        scores = load_or_compute_scores(panel, exp, layout, m, args.split)
        results["methods"][m] = {}
        for n, df in scores.items():
            results["methods"][m][str(n)] = evaluate(df, calib[str(n)], exp, exp.seed)
            r = results["methods"][m][str(n)]
            print(f"[evaluate] {args.split} {m} n={n}: balanced accuracy {r['balanced_accuracy']:.3f} "
                  f"95% CI [{r['balanced_accuracy_ci95'][0]:.3f}, {r['balanced_accuracy_ci95'][1]:.3f}]")
    _write_results(layout, args.split, results)


def cmd_report(args):
    from .report import write_report

    panel, exp, layout = _ctx(args)
    print(write_report(layout, args.split, title="Plant read placement"))


def cmd_gpu_check(args):
    from .encoders import make_encoder
    from .gpucheck import run_check, save
    from .windows import load_windows

    panel, exp, layout = _ctx(args)
    cfg = exp["encoders"][args.encoder]
    amp = args.amp or cfg.get("amp", "none")
    wins, _ = load_windows(layout, args.split)
    rng = np.random.default_rng(exp.seed)
    sample = wins[rng.choice(len(wins), min(args.n, len(wins)), replace=False)]
    total = 0
    for split in ("train", "val", "test"):
        if layout.windows(split).exists():
            total += int(np.load(layout.windows(split))["read_index"].shape[0])
    enc = make_encoder(args.encoder, exp)
    report = run_check(enc, sample, amp, int(cfg.get("batch_size", 64)), total,
                       list(exp["finetune"].get("trainable_regex") or []))
    save(report, layout.root / "gpu_check.json")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        sys.exit("gpu-check found problems: " + "; ".join(report["problems"]))
    print(f"[gpu-check] OK: about {report['estimated_embed_minutes']} min to embed all windows once")


def cmd_list_params(args):
    from .encoders import make_encoder

    panel, exp, layout = _ctx(args)
    enc = make_encoder(args.encoder, exp)
    names = enc.param_names() if hasattr(enc, "param_names") else [n for n, _ in enc.named_parameters()]
    for n in names:
        print(n)
    print(f"[list-params] {len(names)} parameter tensors; use these names for finetune.trainable_regex")


def cmd_check_panel(args):
    panel = load_panel(args.panel)
    warns = panel.validate(strict_pretrain=args.strict)
    counts = pd.Series([s.role for s in panel.species]).value_counts().to_dict()
    print(f"[panel] {panel.name}: {len(panel.species)} species {counts}")
    for w in warns:
        print(f"  warning: {w}")


def cmd_explore(args):
    from .report import explore_distances

    panel, exp, layout = _ctx(args)
    sizes = [int(n) for n in exp["evaluation"]["specimen_sizes"]]
    df = explore_distances(layout, args.split, args.methods, sizes)
    cols = ["method", "n", "known: nearest species right", "relative: nearest genus right",
            "d1 AUROC known vs relative", "d1 AUROC relative vs new", "relatives called known",
            "relatives called relative", "relatives called new"]
    short = ["method", "n", "kn_sp_ok", "rel_ge_ok", "auc_kn_rel", "auc_rel_new", "rel->known", "rel->rel", "rel->new"]
    print(df[cols].set_axis(short, axis=1).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"[explore] written to {layout.results(args.split) / 'EXPLORE_distances.md'}")


def cmd_run_toy(args):
    """Every stage on the toy data set, ending with a frozen preregistration and a test run."""
    steps = [
        ["resolve"], ["fetch"], ["split"], ["simulate", "--workers", "1"], ["windows"],
        ["embed", "--encoder", "kmer"], ["embed", "--encoder", "toycnn"],
        ["finetune", "--encoder", "toycnn", "--name", "toycnn_adapted"],
        ["embed", "--encoder", "toycnn", "--checkpoint", "work/models/toycnn_adapted/encoder.pt", "--name", "toycnn_ft"],
        ["train-head", "--features", "kmer"], ["train-head", "--features", "toycnn"], ["train-head", "--features", "toycnn_ft"],
        ["minimap", "--splits", "val", "test"], ["minimap", "--splits", "val", "test", "--reference", "train"],
        ["dev", "--methods", "kmer_raw", "kmer_head", "toycnn_head", "toycnn_ft_head", "minimap2", "minimap2_train"],
        ["freeze"],
        ["evaluate", "--split", "test", "--methods", "kmer_raw", "kmer_head", "toycnn_head", "toycnn_ft_head", "minimap2",
         "minimap2_train"],
    ]
    for s in steps:
        print(f"\n$ prp {' '.join(s)}")
        main(s + ["--panel", args.panel, "--config", args.config])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prp", description="Plant read placement benchmark")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--panel", default="configs/panel.yaml")
        sp.add_argument("--config", default="configs/experiment.yaml")
        sp.set_defaults(fn=fn)
        return sp

    t = sub.add_parser("toy", help="write a synthetic data set for CPU testing")
    t.add_argument("--out", default="toy")
    t.add_argument("--genome-len", type=int, default=120_000)
    t.set_defaults(fn=cmd_toy)
    add("check-panel", cmd_check_panel, "validate the panel design rules").add_argument("--strict", action="store_true")
    for name, fn, h in [("resolve", cmd_resolve, "choose an NCBI assembly per species"),
                        ("fetch", cmd_fetch, "download genomes and record SHA-256")]:
        add(name, fn, h).add_argument("--datasets", default="datasets")
    add("split", cmd_split, "cut genomes into train/val/test regions").add_argument("--only", nargs="*")
    s = add("simulate", cmd_simulate, "simulate Nanopore reads")
    s.add_argument("--only", nargs="*")
    s.add_argument("--workers", type=int)
    s.add_argument("--backend", choices=["badread", "builtin"])
    add("windows", cmd_windows, "cut reads into model windows")
    e = add("embed", cmd_embed, "embed reads with an encoder")
    e.add_argument("--encoder", required=True)
    e.add_argument("--name")
    e.add_argument("--checkpoint")
    e.add_argument("--batch-size", type=int)
    e.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    e.add_argument("--amp", choices=["none", "fp16", "bf16"], help="override encoders.<name>.amp")
    e.add_argument("--one-process", action="store_true", help="do not spread splits across multiple GPUs")
    add("train-head", cmd_train_head, "train the contrastive head").add_argument("--features", required=True)
    f = add("finetune", cmd_finetune, "adapt encoder layers on noisy reads")
    f.add_argument("--encoder")
    f.add_argument("--name", required=True)
    f.add_argument("--amp", choices=["none", "fp16", "bf16"], help="override encoders.<name>.amp")
    mm = add("minimap", cmd_minimap, "minimap2 baseline")
    mm.add_argument("--splits", nargs="+", default=["val"])
    mm.add_argument("--reference", choices=["full", "train"], default="full",
                    help="full panel genomes (method minimap2) or training blocks only (method minimap2_train)")
    add("dev", cmd_dev, "score + calibrate + evaluate on validation").add_argument("--methods", nargs="+", required=True)
    add("freeze", cmd_freeze, "freeze the preregistration")
    ev = add("evaluate", cmd_evaluate, "evaluate (test requires a frozen preregistration)")
    ev.add_argument("--split", choices=["val", "test"], required=True)
    ev.add_argument("--methods", nargs="+", required=True)
    ev.add_argument("--rerun-reason")
    add("report", cmd_report, "write RESULTS.md and figures").add_argument("--split", choices=["val", "test"], default="val")
    add("list-params", cmd_list_params, "print encoder parameter names").add_argument("--encoder", required=True)
    g = add("gpu-check", cmd_gpu_check, "five-minute pre-flight on the GPU machine")
    g.add_argument("--encoder", default="plantcad")
    g.add_argument("--split", default="val")
    g.add_argument("--n", type=int, default=128)
    g.add_argument("--amp", choices=["none", "fp16", "bf16"])
    ex = add("explore", cmd_explore, "exploratory: where each class goes and why (reads saved scores only)")
    ex.add_argument("--split", choices=["val", "test"], default="val")
    ex.add_argument("--methods", nargs="+", required=True)
    add("run-toy", cmd_run_toy, "run every stage on the toy data set")
    return p


def main(argv=None):
    import os

    from .config import ConfigError
    from .prereg import PreregError

    # Lets PyTorch reuse fragmented GPU memory (the T4 fine-tune ran out with 1 GiB reserved but unused).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_parser().parse_args(argv)
    np.set_printoptions(precision=4, suppress=True)
    try:
        args.fn(args)
    except (PreregError, ConfigError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
