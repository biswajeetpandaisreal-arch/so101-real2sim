"""Closed-loop evaluation of a trained policy in the twin.

Every policy faces the same fixed initial conditions (a paired design):

  seen    the four demonstrated marks
  val     12 random positions, used only to choose the checkpoint
  test    a 7 x 7 grid over the table workspace, each cell labelled
            interp  inside the convex hull of the demonstrated marks
            extrap  outside it

Each trial ends in one of: success (object resting in the box for 0.5 s),
no_contact (object never moved), pushed (moved but never lifted 2 cm) or
not_delivered (lifted, not in the box).

    python scripts/evaluate.py --run outputs/train/gen200_s0 --out results/eval/gen200_s0
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from real2sim.data import TASK, load_episodes
from real2sim.sim import FPS, SO101Env, is_interpolation, read_layout, train_hull, workspace
from real2sim.stats import wilson

GRID_N = 7
LIFTED, MOVED = 0.02, 0.005                         # m
EPISODE_SECONDS = 25.0                              # demonstrations take 9-20 s
OUTCOMES = ("success", "no_contact", "pushed", "not_delivered")


def position_sets(n_val=12, seed=1234):
    layout = read_layout()
    hull = train_hull(layout)
    seen = [("seen", s["xy"]) for s in layout.values() if s["role"] == "train"]
    x0, x1, y0, y1 = workspace(layout)
    test = [("interp" if is_interpolation(xy, hull) else "extrap", xy)
            for xy in (np.array([x, y]) for x in np.linspace(x0, x1, GRID_N) for y in np.linspace(y0, y1, GRID_N))]
    rng = np.random.default_rng(seed)
    taken, val = [xy for _, xy in seen + test], []
    while len(val) < n_val:                                       # off-grid, away from every test cell
        xy = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if min(np.linalg.norm(xy - t) for t in taken) > 0.01:
            val.append(("val", xy))
            taken.append(xy)
    return {"seen": seen, "val": val, "test": test}


def trial_conditions(positions, trials, seed):
    """Per trial: set, cell, a few mm of position jitter and a yaw. Identical for every policy."""
    rng = np.random.default_rng(seed)
    return [{"set": name, "cell": i, "trial": k, "xy": xy + rng.uniform(-0.003, 0.003, 2),
             "yaw": float(rng.uniform(-0.2, 0.2))}
            for i, (name, xy) in enumerate(positions) for k in range(trials)]


def load_policy(path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    policy = get_policy_class(PreTrainedConfig.from_pretrained(path).type).from_pretrained(path)
    policy.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(path))
    return policy, pre, post


def run_trial(env, policy, pre, post, xy, yaw, q0, cameras) -> str:
    env.reset(xy, yaw, q0_lerobot=q0)
    policy.reset()
    device = next(policy.parameters()).device
    start = env.object_pos()
    max_lift = max_move = 0.0
    held = 0
    queue = getattr(policy, "_action_queue", None)                # ACT: render only when the queue is empty
    for _ in range(int(EPISODE_SECONDS * FPS)):
        batch = None
        if queue is None or not queue:
            obs = {"observation.state": torch.tensor(env.state_lerobot(), dtype=torch.float32)[None], "task": [TASK]}
            for c in cameras:
                obs[f"observation.images.{c}"] = torch.from_numpy(env.render(c)).permute(2, 0, 1).float()[None] / 255
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in pre(obs).items()}
        with torch.inference_mode():
            env.step(post(policy.select_action(batch)).squeeze(0).cpu().numpy())
        p = env.object_pos()
        max_lift = max(max_lift, p[2] - start[2])
        max_move = max(max_move, float(np.linalg.norm(p[:2] - start[:2])))
        held = held + 1 if env.success() else 0
        if held >= FPS // 2:
            return "success"
    if env.success():
        return "success"
    if max_lift >= LIFTED:
        return "not_delivered"
    return "pushed" if max_move >= MOVED else "no_contact"


def evaluate(path, conditions, q0, cameras) -> list[dict]:
    env = SO101Env()
    policy, pre, post = load_policy(path)
    rows = [{"set": c["set"], "cell": c["cell"], "trial": c["trial"], "x": c["xy"][0], "y": c["xy"][1],
             "yaw": c["yaw"], "outcome": (o := run_trial(env, policy, pre, post, c["xy"], c["yaw"], q0, cameras)),
             "success": o == "success"} for c in conditions]
    env.close()
    return rows


def summarise(rows) -> dict:
    out = {}
    for s in sorted({r["set"] for r in rows}):
        k, n = sum(r["success"] for r in rows if r["set"] == s), sum(r["set"] == s for r in rows)
        out[s] = {"k": k, "n": n, "rate": round(100 * k / n, 1), "wilson95": [round(v, 1) for v in wilson(k, n)]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", help="training output dir: choose the checkpoint on val, then test it")
    ap.add_argument("--policy", help="evaluate this checkpoint (no selection)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=2, help="trials per test position")
    ap.add_argument("--val-trials", type=int, default=2)
    ap.add_argument("--cameras", nargs="+", default=["top", "wrist_cam"])
    ap.add_argument("--start-from", default="data/datasets/real",
                    help="dataset whose median first state is the start pose")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q0 = np.median([e["state"][0] for e in load_episodes(Path(a.start_from), "local/so101_sim")], axis=0)
    sets = position_sets()
    summary = {"start_pose": q0.tolist()}
    if a.run:
        val = trial_conditions(sets["val"], a.val_trials, a.seed + 1)
        ckpts = sorted((p for p in (Path(a.run) / "checkpoints").iterdir() if p.name.isdigit()),
                       key=lambda p: int(p.name))
        summary["selection"] = []
        for ck in ckpts:
            k = sum(r["success"] for r in evaluate(ck / "pretrained_model", val, q0, a.cameras))
            summary["selection"].append({"checkpoint": ck.name, "val_success": k, "val_n": len(val)})
            print(f"checkpoint {ck.name}: val {k}/{len(val)}", flush=True)
        best = max(summary["selection"], key=lambda s: (s["val_success"], int(s["checkpoint"])))
        summary["selected"] = best["checkpoint"]
        path = Path(a.run) / "checkpoints" / best["checkpoint"] / "pretrained_model"
    else:
        path = Path(a.policy)
    rows = evaluate(path, trial_conditions(sets["seen"] + sets["test"], a.trials, a.seed), q0, a.cameras)
    summary.update(policy=str(path), results=summarise(rows),
                   outcomes={o: sum(r["outcome"] == o for r in rows if r["set"] != "seen") for o in OUTCOMES})
    with open(out / "trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for s, r in summary["results"].items():
        print(f"{s:7s}: {r['k']}/{r['n']} = {r['rate']}% (95% CI {r['wilson95'][0]:.0f}-{r['wilson95'][1]:.0f})")


if __name__ == "__main__":
    main()
