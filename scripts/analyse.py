"""Every number and figure in the README, from the saved results.

Reads results/eval/*/ (policy evaluations), the generation logs, the actuator
identification report and the real-arm execution results, and writes
results/RESULTS.md plus figures in results/figures/.

    python scripts/analyse.py
"""

import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.layout import PAN_AXIS_X, read_layout  # noqa: E402
from scripts.evaluate import wilson  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
MAIN = ("real", "gen200", "naive200")
ABL = ("gen25", "gen100", "gen200", "nofilter200", "reallong", "gen200long")
SETS = ("seen", "interp", "extrap")
NAMES = {"real": "real only", "gen200": "real + 200 generated", "naive200": "real + 200 naive-shift",
         "gen100": "real + 100 generated", "gen25": "real + 25 generated",
         "nofilter200": "real + 200 generated, no physics filter",
         "reallong": "real only, 2x steps (post-hoc)", "gen200long": "real + 200 generated, 2x steps (post-hoc)"}


def load_trials():
    runs = defaultdict(dict)                       # cond -> seed -> rows
    for d in sorted(glob.glob(str(RES / "eval" / "*_s*"))):
        name = Path(d).name
        cond, seed = name.rsplit("_s", 1)
        f = Path(d) / "trials.csv"
        if f.exists():
            with open(f) as fh:
                runs[cond][int(seed)] = [{**r, "success": r["success"] == "True", "cell": int(r["cell"]),
                                          "trial": int(r["trial"]), "x": float(r["x"]), "y": float(r["y"])}
                                         for r in csv.DictReader(fh)]
    return runs


def rate(rows, s):
    sel = [r for r in rows if r["set"] == s]
    return sum(r["success"] for r in sel), len(sel)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / max(n, 1):.0f}% [{lo:.0f}, {hi:.0f}]"


def mcnemar(a, b):
    """Exact McNemar on paired trials (same seed index, cell and trial)."""
    ka = {(r["seed"], r["set"], r["cell"], r["trial"]): r["success"] for r in a}
    kb = {(r["seed"], r["set"], r["cell"], r["trial"]): r["success"] for r in b}
    keys = ka.keys() & kb.keys()
    b01 = sum(not ka[k] and kb[k] for k in keys)
    b10 = sum(ka[k] and not kb[k] for k in keys)
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return b01, b10, p


def policy_section(runs, out):
    out.append("## Policies (ACT, closed loop in simulation)\n")
    out.append("Success with 95% Wilson intervals, pooled over training seeds. "
               "Checkpoints were chosen on validation positions only.\n")
    out.append("| condition | seeds | seen spots | interpolation | extrapolation |")
    out.append("|---|---|---|---|---|")
    pooled = {}
    for c in MAIN + tuple(x for x in ABL if x not in MAIN):
        if c not in runs:
            continue
        rows = [dict(r, seed=s) for s, rr in runs[c].items() for r in rr]
        pooled[c] = rows
        out.append(f"| {NAMES[c]} | {len(runs[c])} | " + " | ".join(fmt(*rate(rows, s)) for s in SETS) + " |")
    out.append("\n**Seed spread** (success % per training seed):\n")
    for c in MAIN:
        if c in runs:
            per = {s: [100 * rate(runs[c][k], s)[0] / max(rate(runs[c][k], s)[1], 1) for k in sorted(runs[c])]
                   for s in SETS}
            out.append(f"- {NAMES[c]}: " + "; ".join(f"{s} " + ", ".join(f"{v:.0f}" for v in per[s]) for s in SETS))
    out.append("\n**Tests** (pooled over seeds; Fisher's exact on the 2x2 success tables, and exact "
               "McNemar on paired trials, since every policy faced identical initial conditions):\n")
    out.append("| comparison | set | Fisher p | McNemar (only B wins / only A wins, p) |")
    out.append("|---|---|---|---|")
    for a, b in (("real", "gen200"), ("real", "naive200"), ("naive200", "gen200")):
        if a in pooled and b in pooled:
            for s in ("interp", "extrap"):
                ka, na = rate(pooled[a], s)
                kb, nb = rate(pooled[b], s)
                p = fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
                pa = [r for r in pooled[a] if r["set"] == s]
                pb = [r for r in pooled[b] if r["set"] == s]
                w, l, pm = mcnemar(pa, pb)
                out.append(f"| {NAMES[a]} (A) vs {NAMES[b]} (B) | {s} | {p:.3g} | {w} / {l}, p = {pm:.3g} |")
    out.append("\n**Failure types** (interpolation + extrapolation trials, pooled):\n")
    out.append("| condition | success | no contact | pushed | lifted, not delivered |")
    out.append("|---|---|---|---|---|")
    for c, rows in pooled.items():
        rr = [r for r in rows if r["set"] != "seen"]
        cnt = {o: sum(r["outcome"] == o for r in rr) for o in ("success", "no_contact", "pushed", "not_delivered")}
        out.append(f"| {NAMES[c]} | " + " | ".join(str(cnt[o]) for o in cnt) + " |")
    return pooled


def heatmaps(pooled, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layout = read_layout()
    conds = [c for c in MAIN if c in pooled]
    if not conds:
        return
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 4.4), squeeze=False)
    for ax, c in zip(axes[0], conds):
        cells = defaultdict(list)
        for r in pooled[c]:
            if r["set"] in ("interp", "extrap"):
                cells[r["cell"]].append(r)
        xs = sorted({round(np.mean([r["x"] for r in v]), 3) for v in cells.values()})
        pts = np.array([[100 * (np.mean([r["x"] for r in v]) - PAN_AXIS_X), 100 * np.mean([r["y"] for r in v]),
                         np.mean([r["success"] for r in v])] for v in cells.values()])
        sc = ax.scatter(-pts[:, 1], pts[:, 0], c=pts[:, 2], cmap="viridis", vmin=0, vmax=1, s=260, marker="s")
        for name, s in layout.items():
            fwd, left = 100 * (s["xy"][0] - PAN_AXIS_X), 100 * s["xy"][1]
            if s["role"] == "basket":
                ax.add_patch(plt.Rectangle((-left - 4.2, fwd - 7.9), 8.4, 15.8, fill=False, ec="k", lw=1))
                continue
            style = {"train": ("o", "white"), "interp": ("^", "orange"), "extrap": ("v", "red")}[s["role"]]
            ax.scatter(-left, fwd, marker=style[0], s=90, facecolor=style[1], edgecolor="k", zorder=3)
            ax.annotate(name, (-left, fwd), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_title(NAMES[c], fontsize=10)
        ax.set_xlabel("right of the robot (cm)")
        ax.set_aspect("equal")
        del xs
    axes[0][0].set_ylabel("forward from the pan axis (cm)")
    fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8, label="success rate")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "success_heatmaps.png", dpi=130, bbox_inches="tight")
    out.append("\n![success over the table](figures/success_heatmaps.png)\n")
    out.append("White circles: demonstrated marks; orange/red triangles: the interpolation/extrapolation "
               "marks used for the real-arm test; rectangle: the box.\n")


def generation_section(out):
    out.append("## Demonstration generation (physics success of generated trajectories)\n")
    out.append("| variant | success |")
    out.append("|---|---|")
    for label, path in (("displacement finished before the final descent (method)", "outputs/gen_timing_before"),
                        ("displacement ramped during the descent (ablation)", "outputs/gen_timing_during"),
                        ("naive shift: source motion unchanged (negative control)", "data/datasets/naive200"),
                        ("method, as rendered for training", "data/datasets/gen200")):
        f = ROOT / path / "generation_log.json"
        if f.exists():
            log = [r for r in json.loads(f.read_text()) if r["kind"] != "replay"]
            out.append(f"| {label} | {fmt(sum(r['success'] for r in log), len(log))} |")
    out.append("")


def twin_section(out):
    out.append("## Digital twin\n")
    f = RES / "sysid.json"
    if f.exists():
        rep = json.loads(f.read_text())
        out.append("Actuator identification (200 Hz servo excitation; validation on held-out teleoperation):\n")
        out.append("| joint | excitation RMSE default → fitted (deg) | held-out teleop RMSE default → fitted (deg) "
                   "| fitted kp (Menagerie 998) | steady-state error after 1-2 deg steps (deg) |")
        out.append("|---|---|---|---|---|")
        for j, e in rep["excitation"].items():
            v = rep.get("validation_rmse_deg", {})
            db = rep["dead_band_deg"].get(j, {})
            small = np.mean([db[k] for k in db if k in ("step 1", "step 2")]) if db else float("nan")
            out.append(f"| {j} | {e['rmse_default_deg']} → {e['rmse_fitted_deg']} | "
                       f"{v.get('default', {}).get(j, '-')} → {v.get('fitted', {}).get(j, '-')} | "
                       f"{e['fitted']['kp']:.0f} | {small:.2f} |")
        out.append("")
    f = ROOT / "data" / "so101_real" / "episodes.csv"
    if f.exists():
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        out.append("Object position recovered from each real grasp (forward kinematics at the jaw stall), "
                   "per mark: spread across demonstrations (cm):\n")
        for m in sorted({r["spot"] for r in rows}):
            p = np.array([[float(r["x"]), float(r["y"])] for r in rows if r["spot"] == m]) * 100
            out.append(f"- mark {m}: n={len(p)}, range {np.ptp(p[:, 0]):.1f} cm forward x {np.ptp(p[:, 1]):.1f} cm sideways")
        out.append("")


def real_section(out):
    files = sorted(glob.glob(str(RES / "real_execution" / "results_*.json")))
    if not files:
        return
    rows = [r for f in files for r in json.loads(Path(f).read_text()) if r["outcome"] != "no_sim_plan"]
    out.append("## Generated trajectories executed open-loop on the real arm\n")
    out.append("| mark | role | success |")
    out.append("|---|---|---|")
    for m in sorted({r["spot"] for r in rows}):
        rr = [r for r in rows if r["spot"] == m]
        out.append(f"| {m} | {rr[0]['role']} | {fmt(sum(r['outcome'] == 'success' for r in rr), len(rr))} |")
    out.append("")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = ["# Results\n", "_Generated by `scripts/analyse.py`; do not edit by hand._\n"]
    twin_section(out)
    real_section(out)
    generation_section(out)
    runs = load_trials()
    if runs:
        pooled = policy_section(runs, out)
        heatmaps(pooled, out)
    RES.mkdir(exist_ok=True)
    (RES / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
