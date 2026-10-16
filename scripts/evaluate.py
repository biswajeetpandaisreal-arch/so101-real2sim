"""Closed-loop evaluation of trained policies in simulation.

Every policy is tested on the same, fixed initial conditions (paired design):

  seen    the four demonstrated cube spots (config/layout.csv, role=train)
  val     12 random positions, used ONLY to choose the checkpoint
  test    a 7 x 7 grid over the table workspace, each cell labelled
            interp  inside the convex hull of the demonstrated spots
            extrap  outside it

Each trial records the outcome and, if it failed, how:
  no_contact      the cube never moved
  pushed          the cube was moved but never lifted 2 cm
  not_delivered   the cube was lifted but did not end in the basket

    # pick the checkpoint on val, then evaluate it on seen + test
    python scripts/evaluate.py --run outputs/train/gen200_s0 --out results/eval/gen200_s0
"""

import argparse
import csv
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.dataset_io import TASK  # noqa: E402
from real2sim.layout import is_interpolation, read_layout, train_hull, workspace  # noqa: E402
from real2sim.sim_env import FPS, SO101Env  # noqa: E402

GRID_N = 7                                                        # 7 x 7 test grid over the workspace
START = np.array([0.0, -100.0, 97.0, 75.0, 0.0, 10.0])            # default start; see --start-from
LIFT, MOVED = 0.02, 0.005


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))


def position_sets(layout=None, n_val=12, seed=1234):
    layout = layout or read_layout()
    hull = train_hull(layout)
    seen = [("seen", s["xy"]) for s in layout.values() if s["role"] == "train"]
    x0, x1, y0, y1 = workspace(layout)
    gx, gy = np.linspace(x0, x1, GRID_N), np.linspace(y0, y1, GRID_N)
    test = [("interp" if is_interpolation(xy, hull) else "extrap", xy)
            for xy in (np.array([x, y]) for x in gx for y in gy)]
    rng = np.random.default_rng(seed)
    taken = [xy for _, xy in seen + test]
    val = []
    while len(val) < n_val:                                        # off-grid, away from test cells
        xy = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if min(np.linalg.norm(xy - t) for t in taken) > 0.01:
            val.append(("val", xy))
            taken.append(xy)
    return {"seen": seen, "val": val, "test": test}


def trial_conditions(positions, trials, seed):
    """Fixed (set, xy, yaw) list; identical for every policy evaluated."""
    rng = np.random.default_rng(seed)
    out = []
    for i, (name, xy) in enumerate(positions):
        for k in range(trials):
            out.append({"set": name, "cell": i, "trial": k,
                        "xy": xy + rng.uniform(-0.003, 0.003, 2), "yaw": float(rng.uniform(-0.2, 0.2))})
    return out


def load_policy(path):
    """Any LeRobot policy checkpoint; camera renames used in training are applied
    by the saved preprocessor."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    cfg = PreTrainedConfig.from_pretrained(path)
    policy = get_policy_class(cfg.type).from_pretrained(path)
    policy.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(path))
    return policy, pre, post


def run_trial(env, policy, pre, post, xy, yaw, q0, cameras, seconds=25.0):   # demos last 9-20 s
    env.reset(xy, yaw, q0_lerobot=q0)
    policy.reset()
    dev = next(policy.parameters()).device
    c0 = env.cube_pos()
    max_lift = max_move = 0.0
    held = 0
    queue = getattr(policy, "_action_queue", None)                 # ACT: observe only when it empties
    for _ in range(int(seconds * FPS)):
        batch = None
        if queue is None or len(queue) == 0:
            obs = {"observation.state": torch.tensor(env.state_lerobot(), dtype=torch.float32)[None],
                   "task": [TASK]}
            for c in cameras:
                obs[f"observation.images.{c}"] = (
                    torch.from_numpy(env.render(c)).permute(2, 0, 1).float()[None] / 255.0)
            batch = pre(obs)
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.inference_mode():
            act = post(policy.select_action(batch))
        env.step(act.squeeze(0).cpu().numpy())
        c = env.cube_pos()
        max_lift = max(max_lift, c[2] - c0[2])
        max_move = max(max_move, float(np.linalg.norm(c[:2] - c0[:2])))
        held = held + 1 if env.success() else 0
        if held >= FPS // 2:                                       # resting in the basket 0.5 s
            return "success"
    if env.success():
        return "success"
    if max_lift >= LIFT:
        return "not_delivered"
    return "pushed" if max_move >= MOVED else "no_contact"


def evaluate(path, conditions, q0, cameras, label=""):
    env = SO101Env()
    policy, pre, post = load_policy(path)
    rows = []
    for c in conditions:
        out = run_trial(env, policy, pre, post, c["xy"], c["yaw"], q0, cameras)
        rows.append({"set": c["set"], "cell": c["cell"], "trial": c["trial"], "x": c["xy"][0],
                     "y": c["xy"][1], "yaw": c["yaw"], "outcome": out, "success": out == "success"})
    env.close()
    return rows


def summarise(rows):
    out = {}
    for s in sorted({r["set"] for r in rows}):
        k = sum(r["success"] for r in rows if r["set"] == s)
        n = sum(r["set"] == s for r in rows)
        out[s] = {"k": k, "n": n, "rate": round(100 * k / n, 1), "wilson95": [round(v, 1) for v in wilson(k, n)]}
    return out


def checkpoints(run):
    ck = Path(run) / "checkpoints"
    return sorted((p / "pretrained_model" for p in ck.iterdir() if p.name.isdigit()), key=lambda p: int(p.parent.name))


def start_pose(start_from):
    if not start_from:
        return START
    from real2sim.dataset_io import load_episodes
    return np.median([e["state"][0] for e in load_episodes(Path(start_from))], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="training output dir: select checkpoint on val, then test")
    ap.add_argument("--policy", help="evaluate one checkpoint (no selection)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=2, help="trials per test cell")
    ap.add_argument("--val-trials", type=int, default=2)
    ap.add_argument("--cameras", nargs="+", default=["top", "wrist_cam"])
    ap.add_argument("--start-from", default=None, help="dataset whose median first state is the start pose")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q0 = start_pose(a.start_from)
    sets = position_sets()
    val = trial_conditions(sets["val"], a.val_trials, a.seed + 1)
    test = trial_conditions(sets["seen"] + sets["test"], a.trials, a.seed)
    summary = {"start_pose": q0.tolist()}
    if a.run:
        sel = []
        for ck in checkpoints(a.run):
            k = sum(r["success"] for r in evaluate(ck, val, q0, a.cameras))
            sel.append({"checkpoint": ck.parent.name, "val_success": k, "val_n": len(val)})
            print(f"checkpoint {ck.parent.name}: val {k}/{len(val)}", flush=True)
        best = max(sel, key=lambda s: (s["val_success"], int(s["checkpoint"])))
        summary["selection"] = sel
        summary["selected"] = best["checkpoint"]
        path = Path(a.run) / "checkpoints" / best["checkpoint"] / "pretrained_model"
    else:
        path = Path(a.policy)
    rows = evaluate(path, test, q0, a.cameras)
    summary["policy"] = str(path)
    summary["results"] = summarise(rows)
    summary["outcomes"] = {o: sum(r["outcome"] == o for r in rows if r["set"] != "seen")
                           for o in ("success", "no_contact", "pushed", "not_delivered")}
    with open(out / "trials.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for s, r in summary["results"].items():
        print(f"{s:7s}: {r['k']}/{r['n']} = {r['rate']}% (95% CI {r['wilson95'][0]:.0f}-{r['wilson95'][1]:.0f})")


if __name__ == "__main__":
    main()
