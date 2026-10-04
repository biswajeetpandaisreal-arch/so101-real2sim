"""Write scripted demonstrations as a stand-in for the real recordings.

Produces data/scripted_test/ (a LeRobot dataset in the SO-101 recording format)
and data/scripted_test/episodes.csv (cube spot per episode), the same pair of
files the real recording session will produce.

    python scripts/make_test_dataset.py --per-spot 3
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim import scripted  # noqa: E402
from real2sim.dataset_io import SimDatasetWriter  # noqa: E402
from real2sim.kinematics import SO101IK  # noqa: E402
from real2sim.sim_env import SO101Env  # noqa: E402

# The marked table spots (m, in the robot base frame) — the same layout the
# real recording uses: 6 spots, 18-32 cm from the base.
SPOTS = {"A": (0.20, 0.12), "B": (0.26, 0.12), "C": (0.30, 0.04),
         "D": (0.24, 0.02), "E": (0.20, -0.04), "F": (0.28, -0.06)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-spot", type=int, default=3)
    ap.add_argument("--out", default="data/scripted_test")
    a = ap.parse_args()
    root = Path(a.out)
    env = SO101Env()
    ik = SO101IK(env)
    rng = np.random.default_rng(0)
    w = SimDatasetWriter(root, "local/so101_scripted_test")
    rows, ep = [], 0
    for spot, xy in SPOTS.items():
        for _ in range(a.per_spot):
            xy_j = np.array(xy) + rng.normal(0, 0.005, 2)     # cube never exactly on the mark
            yaw = rng.uniform(-0.2, 0.2)
            acts = scripted.plan(env, ik, xy_j, yaw, rng)
            ok = scripted.run(env, acts, xy_j, yaw, writer=w)
            w.end_episode(keep=ok)
            if ok:
                rows.append({"episode": ep, "spot": spot, "x": round(xy_j[0], 4),
                             "y": round(xy_j[1], 4), "yaw_deg": round(np.degrees(yaw), 1)})
                ep += 1
            print(f"spot {spot}: {'kept' if ok else 'dropped (failed)'}", flush=True)
    w.close()
    with open(root / "episodes.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=list(rows[0])).writeheader()
        csv.DictWriter(f, fieldnames=list(rows[0])).writerows(rows)
    print(f"wrote {ep} episodes to {root}")


if __name__ == "__main__":
    main()
