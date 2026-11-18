"""Write results tables (Markdown) and figures from evaluation JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import Layout


def _fmt(x, nd=3):
    try:
        if x != x:  # NaN
            return "-"
        return f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def write_report(layout: Layout, split: str, title: str | None = None) -> Path:
    rdir = layout.results(split)
    res = json.loads((rdir / "metrics.json").read_text())
    lines = [f"# {title or 'Results'} ({split} split)", ""]
    if split == "test" and layout.test_marker.exists():
        marker = json.loads(layout.test_marker.read_text())
        lines += [f"Test split first evaluated {marker['first_run']}; deviations recorded: {len(marker.get('deviations', []))}.", ""]
    rows = []
    for method, by_n in res["methods"].items():
        for n, m in by_n.items():
            lo, hi = m["balanced_accuracy_ci95"]
            rows.append({
                "method": method, "reads per specimen": int(n),
                "balanced accuracy [95% CI]": f"{_fmt(m['balanced_accuracy'])} [{_fmt(lo)}, {_fmt(hi)}]",
                "known (species right)": _fmt(m["recall_known"]),
                "relative (genus right)": _fmt(m["recall_relative"]),
                "new": _fmt(m["recall_new"]),
                "AUROC new vs rest": _fmt(m["auroc_new_vs_rest"]),
                "specimens": m["n_specimens"],
            })
    df = pd.DataFrame(rows).sort_values(["reads per specimen", "method"])
    lines += ["## Three-way placement", "",
              "Balanced accuracy is the mean of the three per-class rates. A known specimen counts as correct only "
              "if it is called known *and* assigned the right species; a relative only if called relative *and* "
              "given the right genus; a new specimen if called new. Intervals resample species within each class.",
              "", df.to_markdown(index=False), ""]
    first_method = next(iter(res["methods"]))
    n1 = res["methods"][first_method].get("1")
    if n1:
        lines += ["## Single reads by read identity", ""]
        strat = []
        for method, by_n in res["methods"].items():
            for b in by_n.get("1", {}).get("by_read_identity", []):
                strat.append({"method": method, "identity bin": b["bin"], "reads": b["n"],
                              "balanced accuracy": _fmt(b["balanced_accuracy"])})
        if strat:
            lines += [pd.DataFrame(strat).to_markdown(index=False), ""]
    out = rdir / "RESULTS.md"
    out.write_text("\n".join(lines))
    try:
        _figures(res, rdir)
    except Exception as e:  # figures are a convenience; never fail the report on them
        print(f"[report] figure generation skipped: {e}")
    return out


def _figures(res: dict, rdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, by_n in res["methods"].items():
        ns = sorted(int(n) for n in by_n)
        ys = [by_n[str(n)]["balanced_accuracy"] for n in ns]
        lo = [by_n[str(n)]["balanced_accuracy"] - by_n[str(n)]["balanced_accuracy_ci95"][0] for n in ns]
        hi = [by_n[str(n)]["balanced_accuracy_ci95"][1] - by_n[str(n)]["balanced_accuracy"] for n in ns]
        ax.errorbar(ns, ys, yerr=[lo, hi], marker="o", capsize=3, label=method)
    ax.set_xscale("log")
    ax.set_xlabel("reads per specimen")
    ax.set_ylabel("balanced accuracy (3-way)")
    ax.axhline(1 / 3, color="grey", lw=0.8, ls=":")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(rdir / "balanced_accuracy_vs_reads.png", dpi=150)
    plt.close(fig)


def explore_distances(layout: Layout, split: str, methods: list[str], sizes: list[int]) -> pd.DataFrame:
    """Exploratory, not a preregistered analysis: why does a method lose a class?

    Reads the specimen scores already written by `prp dev` / `prp evaluate` (never scores anything new)
    and separates two failure modes: the nearest prototype is wrong (an embedding problem), or it is right
    but the distance d1 does not separate the classes, so no threshold pair can (a decision problem).
    """
    from .metrics import _auc
    from .placement import decide, load_calibration

    rows = []
    for method in methods:
        calib = load_calibration(layout, method)
        for n in sizes:
            path = layout.scores(method, split, n)
            if not path.exists():
                raise FileNotFoundError(f"{path} not found: run `prp dev` (val) or `prp evaluate` (test) first")
            df = pd.read_csv(path, sep="\t")
            kn, rel, new = (df[df.truth == c] for c in ("known", "relative", "new"))
            c = calib[str(n)]
            called = pd.Series(decide(rel.d1.to_numpy(), c["t1"], c["t2"])).value_counts(normalize=True)
            both = pd.concat([kn, rel])
            rows.append({
                "method": method, "n": n,
                "known: nearest species right": (kn.nearest_species == kn.species).mean(),
                "relative: nearest genus right": (rel.nearest_genus == rel.genus).mean(),
                "d1 AUROC known vs relative": _auc((both.truth == "relative").to_numpy().astype(int), both.d1.to_numpy()),
                "d1 AUROC relative vs new": _auc((pd.concat([rel, new]).truth == "new").to_numpy().astype(int),
                                                 pd.concat([rel, new]).d1.to_numpy()),
                "relatives called known": called.get("known", 0.0),
                "relatives called relative": called.get("relative", 0.0),
                "relatives called new": called.get("new", 0.0),
                "median d1 known": kn.d1.median(), "median d1 relative": rel.d1.median(), "median d1 new": new.d1.median(),
            })
    df = pd.DataFrame(rows)
    out = layout.results(split) / "EXPLORE_distances.md"
    out.write_text(
        f"# Exploratory: where each class goes ({split} split)\n\n"
        "Not a preregistered analysis. 'Nearest ... right' ignores the thresholds; the AUROCs say how well "
        "the distance to the nearest prototype alone separates the classes (0.5 = not at all, "
        "below 0.5 = the second class sits closer than the first); 'relatives "
        "called ...' uses the calibrated thresholds for that specimen size.\n\n"
        + df.to_markdown(index=False, floatfmt=".3f") + "\n")
    return df
