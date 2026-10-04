"""Replay recorded SO-101 episodes in simulation.

Reads a LeRobot dataset (real `lerobot-record` output or a simulated one) and
its episodes.csv (where the cube was in each episode), drives the simulated
SO-101 with the recorded joint commands, and reports which episodes still
succeed in physics. Optionally re-renders the successful ones into a simulated
dataset that a policy can be trained on.

    python scripts/replay.py --data data/so101_real --repo-id <hf_user>/so101_pick_place
    python scripts/replay.py --data data/scripted_test --out data/sim_from_real
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.dataset_io import SimDatasetWriter, load_episodes  # noqa: E402
from real2sim.sim_env import SO101Env  # noqa: E402


def read_spots(path: Path) -> dict[int, dict]:
    with open(path, newline="") as f:
        return {int(r["episode"]): {"xy": (float(r["x"]), float(r["y"])),
                                    "yaw": np.radians(float(r.get("yaw_deg") or 0.0)),
                                    "spot": r.get("spot", "")} for r in csv.DictReader(f)}


def replay(env, ep, spot, writer=None, use="action"):
    traj = ep[use]
    env.reset(spot["xy"], spot["yaw"], q0_lerobot=ep["state"][0])
    max_lift = 0.0
    for a in traj:
        if writer is not None:
            writer.add(env, a)
        env.step(a)
        max_lift = max(max_lift, env.cube_pos()[2])
    return env.success(), max_lift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="LeRobot dataset folder")
    ap.add_argument("--repo-id", default="local/so101_real")
    ap.add_argument("--spots", default=None, help="episodes.csv (default: <data>/episodes.csv)")
    ap.add_argument("--out", default=None, help="write successful replays as a sim dataset here")
    a = ap.parse_args()
    data = Path(a.data)
    spots = read_spots(Path(a.spots) if a.spots else data / "episodes.csv")
    eps = load_episodes(data, a.repo_id)
    env = SO101Env()
    writer = SimDatasetWriter(Path(a.out), "local/so101_sim_replay") if a.out else None
    results = []
    for ep in eps:
        sp = spots.get(ep["episode"])
        if sp is None:
            print(f"episode {ep['episode']}: no cube position in episodes.csv, skipped")
            continue
        ok, lift = replay(env, ep, sp, writer)
        if writer is not None:
            writer.end_episode(keep=ok)
        results.append(ok)
        status = "success" if ok else ("grasped, missed basket" if lift > 0.07 else "no grasp")
        print(f"episode {ep['episode']:3d} spot {sp['spot']}: {status}")
    if writer is not None:
        writer.close()
    print(f"\n{sum(results)}/{len(results)} episodes succeed when replayed in simulation")


if __name__ == "__main__":
    main()
