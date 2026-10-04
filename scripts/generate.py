"""Turn recorded demonstrations into a larger, position-diverse simulated dataset.

For each source episode, samples new cube poses across the workspace, generates
a trajectory with real2sim.generate, runs it in physics, and keeps it only if
the cube lands in the basket. Kept episodes are rendered into a LeRobot dataset
(sim cameras + joint state/action) for policy training. A JSON log records the
source episode and cube pose of every attempt.

    python scripts/generate.py --data data/so101_real --repo-id <id> --per-demo 10 \\
        --out data/sim_generated
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.dataset_io import SimDatasetWriter, load_episodes  # noqa: E402
from real2sim.generate import generate  # noqa: E402
from real2sim.kinematics import SO101IK  # noqa: E402
from real2sim.sim_env import SO101Env  # noqa: E402
from scripts.replay import read_spots, replay  # noqa: E402

# Cube placement region for generated episodes (robot base frame, m)
X_RANGE, Y_RANGE, YAW_RANGE = (0.17, 0.32), (-0.08, 0.15), (-0.4, 0.4)


def sample_pose(rng):
    return (np.array([rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)]),
            float(rng.uniform(*YAW_RANGE)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--repo-id", default="local/so101_real")
    ap.add_argument("--spots", default=None)
    ap.add_argument("--per-demo", type=int, default=10, help="generation attempts per source")
    ap.add_argument("--include-replays", action="store_true",
                    help="also keep the successful replays of the source episodes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data = Path(a.data)
    spots = read_spots(Path(a.spots) if a.spots else data / "episodes.csv")
    eps = [e for e in load_episodes(data, a.repo_id) if e["episode"] in spots]
    env, rng = SO101Env(), np.random.default_rng(a.seed)
    ik = SO101IK(env)
    w = SimDatasetWriter(Path(a.out), "local/so101_sim_generated")
    log, kept = [], 0
    for ep in eps:
        sp = spots[ep["episode"]]
        if a.include_replays:
            ok, _ = replay(env, ep, sp, writer=w)
            w.end_episode(keep=ok)
            kept += ok
            log.append({"source": ep["episode"], "kind": "replay", "x": sp["xy"][0],
                        "y": sp["xy"][1], "yaw": sp["yaw"], "success": bool(ok)})
        for _ in range(a.per_demo):
            xy, yaw = sample_pose(rng)
            try:
                acts = generate(env, ik, ep["action"], sp["xy"], sp["yaw"], xy, yaw)
            except ValueError as err:                 # e.g. gripper never closes
                print(f"episode {ep['episode']}: {err}")
                break
            env.reset(xy, yaw, q0_lerobot=acts[0])
            for act in acts:
                w.add(env, act)
                env.step(act)
            ok = env.success()
            w.end_episode(keep=ok)
            kept += ok
            log.append({"source": ep["episode"], "kind": "generated", "x": float(xy[0]),
                        "y": float(xy[1]), "yaw": yaw, "success": bool(ok)})
        print(f"source {ep['episode']}: kept {kept} so far", flush=True)
    w.close()
    Path(a.out, "generation_log.json").write_text(json.dumps(log, indent=1))
    gen = [r for r in log if r["kind"] == "generated"]
    print(f"\n{kept} episodes kept; generation success "
          f"{sum(r['success'] for r in gen)}/{len(gen)}")


if __name__ == "__main__":
    main()
