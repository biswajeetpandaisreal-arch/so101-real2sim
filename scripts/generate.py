"""Turn recorded demonstrations into a larger, position-diverse simulated dataset.

For each source episode, samples new cube poses across the workspace, makes a
trajectory for each, runs it in physics and (by default) keeps it only if the
cube lands in the basket. Kept episodes are rendered into a LeRobot dataset
(sim cameras + joint state/action) for policy training. A JSON log records the
source episode, cube pose, outcome and dataset episode index of every attempt.

Modes (for the controlled comparison and ablations):
  generate  object-centric adaptation (real2sim.generate)          [method]
  naive     the source actions unchanged, cube moved               [negative control]
Options:
  --keep-failures        no physics filter                         [ablation]
  --shift-timing during  displacement ramped during the descent    [ablation]

    python scripts/generate.py --data data/so101_real --total 200 --include-replays \\
        --out data/gen_200
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
from real2sim.layout import read_layout, workspace  # noqa: E402
from real2sim.sim_env import SO101Env  # noqa: E402
from scripts.replay import load_demos, read_spots, replay  # noqa: E402

# Cube placement region for generated episodes: the table layout's workspace
X0, X1, Y0, Y1 = workspace(read_layout())
YAW_RANGE = (-0.4, 0.4)


def sample_pose(rng):
    return (np.array([rng.uniform(X0, X1), rng.uniform(Y0, Y1)]), float(rng.uniform(*YAW_RANGE)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--repo-id", default="local/so101_real")
    ap.add_argument("--spots", default=None)
    ap.add_argument("--total", type=int, default=200, help="generated episodes to keep (0: replays only)")
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--mode", choices=("generate", "naive"), default="generate")
    ap.add_argument("--shift-timing", choices=("before", "during"), default="before")
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--include-replays", action="store_true",
                    help="also keep the successful replays of the source episodes")
    ap.add_argument("--sources", default=None, help="comma-separated spot names to use as sources")
    ap.add_argument("--no-render", action="store_true", help="physics only (generation statistics)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    spots = read_spots(Path(a.spots) if a.spots else data / "episodes.csv")
    eps = load_demos(data, a.repo_id, spots)
    if a.sources:
        keep = set(a.sources.split(","))
        eps = [e for e in eps if spots[e["episode"]]["spot"] in keep]
    env, rng = SO101Env(), np.random.default_rng(a.seed)
    ik = SO101IK(env)
    valid = [e for e in eps if replay(env, e, spots[e["episode"]])[0]]   # sources the twin reproduces
    print(f"{len(valid)}/{len(eps)} source demonstrations succeed when replayed; using those", flush=True)
    eps = valid
    w = None if a.no_render else SimDatasetWriter(out, "local/so101_sim")
    log, n_ep = [], 0

    def finish(keep, row):
        nonlocal n_ep
        row["dataset_episode"] = n_ep if keep and w is not None else None
        if w is not None:
            w.end_episode(keep=keep)
        n_ep += bool(keep) and w is not None
        log.append(row)

    if a.include_replays:
        for ep in eps:
            sp = spots[ep["episode"]]
            ok, _ = replay(env, ep, sp, writer=w)
            finish(ok, {"source": ep["episode"], "spot": sp["spot"], "kind": "replay",
                        "x": sp["xy"][0], "y": sp["xy"][1], "yaw": sp["yaw"], "success": bool(ok)})
    max_attempts = a.max_attempts or a.total * (1 if a.keep_failures or a.mode == "naive" else 3)
    kept = attempts = 0
    while kept < a.total and attempts < max_attempts:
        ep = eps[attempts % len(eps)]                      # cycle through the sources evenly
        sp = spots[ep["episode"]]
        attempts += 1
        xy, yaw = sample_pose(rng)
        if a.mode == "naive":
            acts = ep["action"]
        else:
            try:
                acts = generate(env, ik, ep["action"], sp["xy"], sp["yaw"], xy, yaw, a.shift_timing)
            except ValueError as err:                      # e.g. gripper never closes
                print(f"episode {ep['episode']}: {err}")
                continue
        env.reset(xy, yaw, q0_lerobot=acts[0])
        for act in acts:
            if w is not None:
                w.add(env, act)
            env.step(act)
        ok = env.success()
        keep = ok or a.keep_failures or a.mode == "naive"
        kept += keep
        finish(keep, {"source": ep["episode"], "spot": sp["spot"], "kind": a.mode,
                      "shift_timing": a.shift_timing, "x": float(xy[0]), "y": float(xy[1]),
                      "yaw": yaw, "success": bool(ok)})
        if attempts % 20 == 0:
            print(f"{attempts} attempts, {kept} kept", flush=True)
    if w is not None:
        w.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "generation_log.json").write_text(json.dumps(log, indent=1))
    gen = [r for r in log if r["kind"] != "replay"]
    print(f"\n{n_ep} episodes in dataset; {a.mode} physics success "
          f"{sum(r['success'] for r in gen)}/{len(gen)}")


if __name__ == "__main__":
    main()
