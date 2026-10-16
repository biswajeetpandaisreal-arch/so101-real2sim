"""Write <data>/episodes.csv: every usable recorded demonstration with the
object position measured from its grasp, the nearest table mark, and the
frames that cover exactly one pick-and-place (idle time and any second
attempt cut off). Episodes without a grasp are left out.

    python scripts/make_episode_list.py --data data/so101_real
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.dataset_io import load_episodes  # noqa: E402
from real2sim.demo_analysis import object_position, task_span  # noqa: E402
from real2sim.kinematics import SO101IK  # noqa: E402
from real2sim.layout import read_layout  # noqa: E402
from real2sim.sim_env import SO101Env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/so101_real")
    ap.add_argument("--repo-id", default="local/so101_real")
    a = ap.parse_args()
    env = SO101Env()
    ik = SO101IK(env)
    marks = {k: v["xy"] for k, v in read_layout().items() if v["role"] == "train"}
    rows = []
    for ep in load_episodes(Path(a.data), a.repo_id):
        span = task_span(ep["action"], ep["state"])
        xy, _ = object_position(env, ik, ep["action"], ep["state"])
        if span is None or xy is None:
            continue
        spot = min(marks, key=lambda k: np.linalg.norm(marks[k] - xy))
        rows.append({"episode": ep["episode"], "spot": spot, "x": round(float(xy[0]), 4),
                     "y": round(float(xy[1]), 4), "yaw_deg": 0, "start": span[0], "end": span[1]})
    with open(Path(a.data) / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} demonstrations with a grasp: " +
          ", ".join(f"mark {m}: {sum(r['spot'] == m for r in rows)}" for m in sorted(marks)))


if __name__ == "__main__":
    main()
