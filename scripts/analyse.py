"""Every number and figure in the README, computed from results/ alone.

Inputs (all committed): results/eval/<run>/trials.csv (policy evaluations),
results/generation/*.json (generation logs), results/sysid.json (actuator
identification), results/replay_check.json and results/episodes.csv (the
recorded demonstrations against the twin), results/real_execution/*.json
(generated trajectories on the real arm).
Outputs: results/RESULTS.md and results/figures/success_heatmaps.png.

    python scripts/analyse.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, fisher_exact

from real2sim.sim import read_layout, to_table_cm
from real2sim.stats import fmt_rate

RESULTS = Path(__file__).resolve().parent.parent / "results"
SETS = ("seen", "interp", "extrap")
OUTCOMES = ("success", "no_contact", "pushed", "not_delivered")
NAMES = {"real": "real only", "gen200": "real + 200 generated", "naive200": "real + 200 naive-shift",
         "reallong": "real only, 2x steps (post-hoc)", "gen200long": "real + 200 generated, 2x steps (post-hoc)",
         "gen100": "real + 100 generated", "gen25": "real + 25 generated",
         "nofilter200": "real + 200 generated, no physics filter"}
MAIN = ("real", "gen200", "naive200")
COMPARISONS = (("real", "gen200"), ("real", "naive200"), ("naive200", "gen200"), ("reallong", "gen200long"))


def load_runs() -> dict[str, list[dict]]:
    """condition -> evaluation trials pooled over seeds (each row tagged with its seed)."""
    runs = defaultdict(list)
    for f in sorted(RESULTS.glob("eval/*_s*/trials.csv")):
        cond, seed = f.parent.name.rsplit("_s", 1)
        with open(f) as fh:
            runs[cond] += [{**r, "seed": int(seed), "cell": int(r["cell"]), "trial": int(r["trial"]),
                            "x": float(r["x"]), "y": float(r["y"]), "success": r["success"] == "True"}
                           for r in csv.DictReader(fh)]
    return dict(runs)


def count(rows, s=None):
    sel = [r for r in rows if s is None or r["set"] == s]
    return sum(r["success"] for r in sel), len(sel)


def mcnemar(a, b):
    """Exact McNemar test on trials paired by (seed, cell, trial): every policy faced the same conditions."""
    key = lambda r: (r["seed"], r["cell"], r["trial"])                               # noqa: E731
    sa, sb = {key(r): r["success"] for r in a}, {key(r): r["success"] for r in b}
    shared = sa.keys() & sb.keys()
    only_b, only_a = sum(sb[k] and not sa[k] for k in shared), sum(sa[k] and not sb[k] for k in shared)
    return only_b, only_a, binomtest(only_b, only_b + only_a).pvalue if only_a + only_b else 1.0


def twin_section(out):
    out.append("## Digital twin\n")
    rep = json.loads((RESULTS / "sysid.json").read_text())
    out += ["Actuator identification (200 Hz servo excitation; validation on held-out teleoperation):\n",
            "| joint | excitation RMSE default → fitted (deg) | held-out teleoperation RMSE default → fitted (deg) "
            "| fitted kp (Menagerie: 998) | steady-state error after 1-2 deg steps (deg) |", "|---|---|---|---|---|"]
    val = rep["validation_rmse_deg"]
    for j, e in rep["excitation"].items():
        small = np.mean([v for k, v in rep["dead_band_deg"][j].items() if k in ("step 1", "step 2")])
        out.append(f"| {j} | {e['rmse_default_deg']} → {e['rmse_fitted_deg']} | {val['default'][j]} → "
                   f"{val['fitted'][j]} | {e['fitted']['kp']:.0f} | {small:.2f} |")
    check = json.loads((RESULTS / "replay_check.json").read_text())
    out.append("\nRecorded leader commands replayed in the twin (object at the position measured from the grasp): "
               + "; ".join(f"{k}: {sum(v.values())}/{len(v)} succeed" for k, v in check.items()) + "\n")
    with open(RESULTS / "episodes.csv") as f:
        rows = list(csv.DictReader(f))
    out.append("Object position recovered from each real grasp, spread across demonstrations per mark:\n")
    for m in sorted({r["mark"] for r in rows}):
        p = np.array([to_table_cm((float(r["x"]), float(r["y"]))) for r in rows if r["mark"] == m])
        out.append(f"- mark {m}: {len(p)} demonstrations, "
                   f"{np.ptp(p[:, 0]):.1f} cm forward x {np.ptp(p[:, 1]):.1f} cm sideways")
    out.append("")


def real_section(out):
    rows = [r for f in sorted(RESULTS.glob("real_execution/results_*.json"))
            for r in json.loads(f.read_text()) if r["outcome"] != "no_sim_plan"]
    out += ["## Generated trajectories executed open loop on the real arm\n",
            "| mark | role | in the box |", "|---|---|---|"]
    for m in sorted({r["mark"] for r in rows}):
        rr = [r for r in rows if r["mark"] == m]
        out.append(f"| {m} | {rr[0]['role']} | {fmt_rate(sum(r['outcome'] == 'success' for r in rr), len(rr))} |")
    out.append("")


def generation_section(out):
    out += ["## Demonstration generation (physics success of generated trajectories)\n",
            "| variant | success |", "|---|---|"]
    for label, name in (("displacement finished before the final descent (method)", "timing_before"),
                        ("displacement ramped during the descent (ablation)", "timing_during"),
                        ("naive shift: source commands unchanged (negative control)", "naive200"),
                        ("method, as rendered for training", "gen200")):
        log = [r for r in json.loads((RESULTS / "generation" / f"{name}.json").read_text()) if r["kind"] != "replay"]
        out.append(f"| {label} | {fmt_rate(sum(r['success'] for r in log), len(log))} |")
    out.append("")


def policy_section(runs, out):
    out += ["## Policies (ACT, closed loop in simulation)\n",
            "Success with 95% Wilson intervals, pooled over training seeds; checkpoints chosen on validation "
            "positions only.\n",
            "| condition | seeds | seen marks | interpolation | extrapolation |", "|---|---|---|---|---|"]
    for c in [c for c in NAMES if c in runs]:
        out.append(f"| {NAMES[c]} | {len({r['seed'] for r in runs[c]})} | "
                   + " | ".join(fmt_rate(*count(runs[c], s)) for s in SETS) + " |")
    out.append("\n**Per seed** (success %):\n")
    for c in [c for c in MAIN if c in runs]:
        per_set = []
        for s in SETS:
            rates = []
            for seed in sorted({r["seed"] for r in runs[c]}):
                k, n = count([r for r in runs[c] if r["seed"] == seed], s)
                rates.append(f"{100 * k / n:.0f}")
            per_set.append(f"{s} " + ", ".join(rates))
        out.append(f"- {NAMES[c]}: " + "; ".join(per_set))
    out += ["\n**Tests** (pooled over seeds): Fisher's exact on the success counts, and exact McNemar on trials "
            "paired by seed, position and trial.\n",
            "| A | B | set | Fisher p | McNemar: only B / only A succeeds, p |", "|---|---|---|---|---|"]
    for a, b in [(a, b) for a, b in COMPARISONS if a in runs and b in runs]:
        for s in ("interp", "extrap"):
            (ka, na), (kb, nb) = count(runs[a], s), count(runs[b], s)
            only_b, only_a, p = mcnemar([r for r in runs[a] if r["set"] == s], [r for r in runs[b] if r["set"] == s])
            out.append(f"| {NAMES[a]} | {NAMES[b]} | {s} | {fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]:.3g} | "
                       f"{only_b} / {only_a}, p = {p:.3g} |")
    out += ["\n**How the trials ended** (interpolation + extrapolation, pooled). \"Lifted\" = success + lifted but "
            "not delivered.\n", "| condition | success | no contact | pushed | lifted, not delivered | lifted |",
            "|---|---|---|---|---|---|"]
    for c in [c for c in NAMES if c in runs]:
        rows = [r for r in runs[c] if r["set"] != "seen"]
        n = {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}
        lifted = n["success"] + n["not_delivered"]
        out.append(f"| {NAMES[c]} | " + " | ".join(str(n[o]) for o in OUTCOMES) + f" | {fmt_rate(lifted, len(rows))} |")
    out += ["\n**Exploratory, not pre-registered:** is the object lifted at all (interpolation + extrapolation)?\n",
            "| A | B | lifted A | lifted B | Fisher p |", "|---|---|---|---|---|"]
    for a, b in [(a, b) for a, b in COMPARISONS if a in runs and b in runs]:
        counts = []
        for c in (a, b):
            rows = [r for r in runs[c] if r["set"] != "seen"]
            counts.append((sum(r["outcome"] in ("success", "not_delivered") for r in rows), len(rows)))
        (ka, na), (kb, nb) = counts
        out.append(f"| {NAMES[a]} | {NAMES[b]} | {fmt_rate(ka, na)} | {fmt_rate(kb, nb)} | "
                   f"{fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]:.2g} |")
    out.append("")


def heatmaps(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [c for c in MAIN if c in runs]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 4.6), squeeze=False)
    for ax, c in zip(axes[0], conds):
        cells = defaultdict(list)
        for r in runs[c]:
            if r["set"] != "seen":
                cells[r["cell"]].append(r)
        pts = np.array([[*to_table_cm((np.mean([r["x"] for r in v]), np.mean([r["y"] for r in v]))),
                         np.mean([r["success"] for r in v])] for v in cells.values()])
        sc = ax.scatter(-pts[:, 1], pts[:, 0], c=pts[:, 2], cmap="viridis", vmin=0, vmax=1, s=260, marker="s")
        for name, s in read_layout().items():
            fwd, left = to_table_cm(s["xy"])
            if s["role"] == "box":
                ax.add_patch(plt.Rectangle((-left - 4.2, fwd - 7.9), 8.4, 15.8, fill=False, ec="k", lw=1))
                continue
            marker, color = {"train": ("o", "white"), "interp": ("^", "orange"), "extrap": ("v", "red")}[s["role"]]
            ax.scatter(-left, fwd, marker=marker, s=90, facecolor=color, edgecolor="k", zorder=3)
            ax.annotate(name, (-left, fwd), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set(title=NAMES[c], xlabel="right of the robot (cm)", aspect="equal")
    axes[0][0].set_ylabel("forward from the pan axis (cm)")
    fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8, label="success rate")
    (RESULTS / "figures").mkdir(exist_ok=True)
    fig.savefig(RESULTS / "figures" / "success_heatmaps.png", dpi=130, bbox_inches="tight")
    out += ["![success over the table](figures/success_heatmaps.png)\n",
            "White circles: demonstrated marks; orange/red triangles: the marks used for the real-arm test; "
            "rectangle: the box.\n"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = ["# Results\n", "_Generated by `scripts/analyse.py` from the files in results/._\n"]
    twin_section(out)
    real_section(out)
    generation_section(out)
    if runs := load_runs():
        policy_section(runs, out)
        heatmaps(runs, out)
    (RESULTS / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
