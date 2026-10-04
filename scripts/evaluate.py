"""Closed-loop evaluation of a trained policy in simulation.

Places the cube at each test pose, runs the policy from the observations the
simulator renders (same cameras and joint state as the training data), and
records whether the cube ends in the basket. Two test sets:

  seen    the cube spots used in the real recordings (episodes.csv)
  unseen  a grid of positions across the workspace that no source demonstration
          used — the generalisation test

    python scripts/evaluate.py --policy outputs/train/act_real/checkpoints/last/pretrained_model \\
        --spots data/so101_real/episodes.csv --trials 3 --csv outputs/eval_act_real.csv
"""

import argparse
import csv
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.dataset_io import TASK  # noqa: E402
from real2sim.sim_env import FPS, SO101Env  # noqa: E402
from scripts.replay import read_spots  # noqa: E402

# Unseen test grid: inside the generation region, away from the recorded spots
UNSEEN = [(x, y) for x in (0.18, 0.23, 0.28, 0.32) for y in (-0.07, 0.0, 0.07, 0.14)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load_policy(path):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    policy = ACTPolicy.from_pretrained(path)
    policy.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(path))
    return policy, pre, post


def run_trial(env, policy, pre, post, xy, yaw, q0, cameras, seconds=15.0):
    env.reset(xy, yaw, q0_lerobot=q0)
    policy.reset()
    dev = next(policy.parameters()).device
    held = 0
    for _ in range(int(seconds * FPS)):
        obs = {"observation.state": torch.tensor(env.state_lerobot(), dtype=torch.float32)[None],
               "task": [TASK]}
        for c in cameras:
            img = torch.from_numpy(env.render(c)).permute(2, 0, 1).float()[None] / 255.0
            obs[f"observation.images.{c}"] = img
        with torch.inference_mode():
            batch = pre(obs)
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            act = post(policy.select_action(batch))
        env.step(act.squeeze(0).cpu().numpy())
        held = held + 1 if env.success() else 0
        if held >= FPS // 2:                      # resting in the basket for 0.5 s
            return True
    return env.success()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--spots", required=True, help="episodes.csv of the recorded data (seen set)")
    ap.add_argument("--start-from", default=None,
                    help="LeRobot dataset whose median first state is the start pose")
    ap.add_argument("--trials", type=int, default=3, help="trials per position")
    ap.add_argument("--cameras", nargs="+", default=["top", "wrist_cam"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    env = SO101Env()
    policy, pre, post = load_policy(a.policy)
    if a.start_from:
        from real2sim.dataset_io import load_episodes
        q0 = np.median([e["state"][0] for e in load_episodes(Path(a.start_from))], axis=0)
    else:
        q0 = np.array([0.0, -52.0, 19.0, 95.0, 0.0, 60.0])   # scripted ready pose
    rng = np.random.default_rng(a.seed)
    seen = sorted({(round(s["xy"][0], 3), round(s["xy"][1], 3))
                   for s in read_spots(Path(a.spots)).values()})
    rows = []
    for name, poses in (("seen", seen), ("unseen", UNSEEN)):
        k = n = 0
        for xy in poses:
            for _ in range(a.trials):
                xy_j = np.array(xy) + rng.normal(0, 0.005, 2)
                yaw = float(rng.uniform(-0.2, 0.2))
                ok = run_trial(env, policy, pre, post, xy_j, yaw, q0, a.cameras)
                rows.append({"set": name, "x": xy_j[0], "y": xy_j[1], "yaw": yaw, "success": ok})
                k += ok
                n += 1
        lo, hi = wilson(k, n)
        print(f"{name:7s}: {k}/{n} = {100 * k / max(n, 1):.1f}% (95% CI {lo:.0f}-{hi:.0f})",
              flush=True)
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
