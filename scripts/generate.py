"""Build a simulated training dataset from the recorded demonstrations.

Sources are the listed demonstrations the twin reproduces. For each new object
pose (sampled over the table workspace, cycling through the sources) a
trajectory is produced and run in physics; kept episodes are rendered into a
LeRobot dataset (top and wrist cameras, joint state and command).
generation_log.json (also copied to results/generation/<out name>.json)
records every attempt.

  --mode generate        object-centric adaptation (real2sim.generate)   method
  --mode naive           source commands unchanged, object moved        negative control
  --keep-failures        no physics filter                              ablation
  --shift-timing during  displacement ramped during the descent         ablation
  --include-replays      also keep the sources' own replays             (the "real" data)
  --no-render            physics only, for generation statistics

    python scripts/generate.py --total 200 --include-replays --out data/datasets/gen200
    python scripts/generate.py --total 200 --keep-failures --no-render --out data/datasets/timing_before
"""

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim.dataset_io import SimDatasetWriter
from real2sim.demos import load_demos, read_episode_list, replay
from real2sim.generate import generate
from real2sim.kinematics import SO101IK
from real2sim.layout import read_layout, workspace
from real2sim.sim_env import ROOT, SO101Env

YAW_RANGE = (-0.4, 0.4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/so101_real")
    ap.add_argument("--repo-id", default="local/so101_real")
    ap.add_argument("--episodes", default=str(ROOT / "results" / "episodes.csv"))
    ap.add_argument("--total", type=int, default=200, help="episodes to keep besides replays")
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--mode", choices=("generate", "naive"), default="generate")
    ap.add_argument("--shift-timing", choices=("before", "during"), default="before")
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--include-replays", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    env, rng = SO101Env(), np.random.default_rng(a.seed)
    ik = SO101IK(env)
    x0, x1, y0, y1 = workspace(read_layout())
    demos = load_demos(Path(a.data), a.repo_id, read_episode_list(Path(a.episodes)))
    sources = [d for d in demos if replay(env, d)]
    print(f"{len(sources)}/{len(demos)} demonstrations succeed when replayed; using them as sources", flush=True)
    writer = None if a.no_render else SimDatasetWriter(out, "local/so101_sim")
    log, n_saved = [], 0

    def record(keep, row):
        nonlocal n_saved
        row["dataset_episode"] = n_saved if keep and writer else None
        if writer:
            writer.end_episode(keep)
            n_saved += keep
        log.append(row)

    if a.include_replays:
        for d in sources:
            ok = replay(env, d, writer)
            record(ok, {"source": d["episode"], "mark": d["info"]["mark"], "kind": "replay",
                        "x": float(d["info"]["xy"][0]), "y": float(d["info"]["xy"][1]), "yaw": d["info"]["yaw"],
                        "success": ok})
    max_attempts = a.max_attempts or a.total * (1 if a.keep_failures or a.mode == "naive" else 3)
    kept = attempts = 0
    while kept < a.total and attempts < max_attempts:
        d = sources[attempts % len(sources)]
        attempts += 1
        xy, yaw = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)]), float(rng.uniform(*YAW_RANGE))
        if a.mode == "naive":
            commands = d["action"]
        else:
            commands = generate(env, ik, d["action"], d["info"]["xy"], d["info"]["yaw"], xy, yaw, a.shift_timing)
        env.reset(xy, yaw, q0_lerobot=commands[0])
        for c in commands:
            if writer:
                writer.add(env, c)
            env.step(c)
        ok = env.success()
        keep = ok or a.keep_failures or a.mode == "naive"
        kept += keep
        record(keep, {"source": d["episode"], "mark": d["info"]["mark"], "kind": a.mode,
                      "shift_timing": a.shift_timing, "x": float(xy[0]), "y": float(xy[1]), "yaw": yaw,
                      "success": ok})
        if attempts % 20 == 0:
            print(f"{attempts} attempts, {kept} kept", flush=True)
    if writer:
        writer.close()
    out.mkdir(parents=True, exist_ok=True)
    for path in (out / "generation_log.json", ROOT / "results" / "generation" / f"{out.name}.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(log, indent=1))
    attempts_log = [r for r in log if r["kind"] != "replay"]
    print(f"{n_saved} episodes saved; {a.mode} physics success "
          f"{sum(r['success'] for r in attempts_log)}/{len(attempts_log)}")


if __name__ == "__main__":
    main()
