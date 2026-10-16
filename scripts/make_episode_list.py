"""List the usable recorded demonstrations and check them against the twin.

For every episode with a grasp, writes the object position measured from the
grasp, the nearest table mark and the frames covering exactly one
pick-and-place to results/episodes.csv. Then replays each listed demonstration's
leader commands in the twin, with the default and with the identified
actuators, and records which ones succeed (results/replay_check.json).

    python scripts/make_episode_list.py --data data/so101_real
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from real2sim.dataset_io import load_episodes
from real2sim.demos import load_demos, object_position, read_episode_list, replay, task_span
from real2sim.kinematics import SO101IK
from real2sim.layout import read_layout, to_table_cm
from real2sim.sim_env import ROOT, SO101Env


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/so101_real")
    ap.add_argument("--repo-id", default="local/so101_real")
    ap.add_argument("--out", default=str(ROOT / "results" / "episodes.csv"))
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    env = SO101Env()
    ik = SO101IK(env)
    marks = {k: v["xy"] for k, v in read_layout().items() if v["role"] == "train"}
    rows = []
    for ep in load_episodes(data, a.repo_id):
        span = task_span(ep["action"], ep["state"])
        xy, _ = object_position(env, ik, ep["action"], ep["state"])
        if span is None or xy is None:
            print(f"episode {ep['episode']:2d}: no grasp, left out")
            continue
        mark = min(marks, key=lambda k: np.linalg.norm(marks[k] - xy))
        rows.append({"episode": ep["episode"], "mark": mark, "x": round(float(xy[0]), 4),
                     "y": round(float(xy[1]), 4), "yaw_deg": 0, "start": span[0], "end": span[1]})
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}: {len(rows)} demonstrations with a grasp")

    demos = load_demos(data, a.repo_id, read_episode_list(out))
    twins = {"default actuators": SO101Env(actuators=None), "identified actuators": env}
    ok = {name: [replay(twin, d) for d in demos] for name, twin in twins.items()}
    for d, *results in zip(demos, *ok.values()):
        fwd, left = to_table_cm(d["info"]["xy"])
        print(f"episode {d['episode']:2d} mark {d['info']['mark']} object at {fwd:4.1f} cm fwd {left:5.1f} cm left: "
              + ", ".join(f"{n} {'ok' if r else 'fail'}" for n, r in zip(ok, results)))
    for name, results in ok.items():
        print(f"replay with {name}: {sum(results)}/{len(results)} succeed")
    check = {name: {str(d["episode"]): r for d, r in zip(demos, results)} for name, results in ok.items()}
    (out.parent / "replay_check.json").write_text(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
