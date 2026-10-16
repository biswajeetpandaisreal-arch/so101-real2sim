"""From the real recordings to simulated training data.

episodes  For every recorded episode with a grasp, write the object position
          measured from the grasp, the nearest mark and the frames covering
          exactly one pick-and-place to results/episodes.csv. Replay each
          listed demonstration's leader commands in the twin with default and
          identified actuators and record which succeed (results/replay_check.json).
generate  Build a simulated LeRobot dataset (top and wrist cameras, joint state
          and command) from the demonstrations the twin reproduces. For each new
          object pose (sampled over the table workspace, cycling through the
          sources) a trajectory is produced and run in physics. Every attempt is
          logged to <out>/generation_log.json and results/generation/<out name>.json.

            --mode generate        object-centric adaptation (method)
            --mode naive           source commands unchanged, object moved (negative control)
            --keep-failures        no physics filter (ablation)
            --shift-timing during  displacement ramped during the descent (ablation)
            --include-replays      also keep the sources' own replays (the "real" data)
            --no-render            physics only, for generation statistics

    python scripts/prepare_data.py episodes
    python scripts/prepare_data.py generate --total 0 --include-replays --out data/datasets/real
    python scripts/prepare_data.py generate --total 200 --include-replays --out data/datasets/gen200
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from real2sim.data import (
    SimDatasetWriter,
    load_demos,
    load_episodes,
    object_position,
    read_episode_list,
    replay,
    task_span,
)
from real2sim.generate import generate
from real2sim.kinematics import SO101IK
from real2sim.sim import ROOT, SO101Env, read_layout, to_table_cm, workspace

EPISODES = ROOT / "results" / "episodes.csv"
YAW_RANGE = (-0.4, 0.4)


def episodes(a):
    env = SO101Env()
    ik = SO101IK(env)
    marks = {k: v["xy"] for k, v in read_layout().items() if v["role"] == "train"}
    rows = []
    for ep in load_episodes(Path(a.data), a.repo_id):
        span = task_span(ep["action"], ep["state"])
        xy, _ = object_position(env, ik, ep["action"], ep["state"])
        if span is None or xy is None:
            print(f"episode {ep['episode']:2d}: no grasp, left out")
            continue
        mark = min(marks, key=lambda k: np.linalg.norm(marks[k] - xy))
        rows.append({"episode": ep["episode"], "mark": mark, "x": round(float(xy[0]), 4),
                     "y": round(float(xy[1]), 4), "yaw_deg": 0, "start": span[0], "end": span[1]})
    with open(a.episodes, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {a.episodes}: {len(rows)} demonstrations with a grasp")

    demos = load_demos(Path(a.data), a.repo_id, read_episode_list(Path(a.episodes)))
    twins = {"default actuators": SO101Env(actuators=None), "identified actuators": env}
    ok = {name: [replay(twin, d) for d in demos] for name, twin in twins.items()}
    for d, *results in zip(demos, *ok.values()):
        fwd, left = to_table_cm(d["info"]["xy"])
        print(f"episode {d['episode']:2d} mark {d['info']['mark']} at {fwd:4.1f} cm fwd {left:5.1f} cm left: "
              + ", ".join(f"{n} {'ok' if r else 'fail'}" for n, r in zip(ok, results)))
    for name, results in ok.items():
        print(f"replay with {name}: {sum(results)}/{len(results)} succeed")
    check = {name: {str(d["episode"]): r for d, r in zip(demos, results)} for name, results in ok.items()}
    (Path(a.episodes).parent / "replay_check.json").write_text(json.dumps(check, indent=1))


def generate_dataset(a):
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
    for path in (out / "generation_log.json", ROOT / "results" / "generation" / f"{out.name}.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(log, indent=1))
    attempted = [r for r in log if r["kind"] != "replay"]
    print(f"{n_saved} episodes saved; {a.mode} physics success {sum(r['success'] for r in attempted)}/{len(attempted)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/so101_real", help="the real recordings")
    ap.add_argument("--repo-id", default="local/so101_real")
    ap.add_argument("--episodes", default=str(EPISODES))
    sub = ap.add_subparsers(required=True)
    sub.add_parser("episodes", help="list and check the recorded demonstrations").set_defaults(run=episodes)
    p = sub.add_parser("generate", help="build a simulated training dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--total", type=int, default=200, help="episodes to keep besides replays")
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--mode", choices=("generate", "naive"), default="generate")
    p.add_argument("--shift-timing", choices=("before", "during"), default="before")
    p.add_argument("--keep-failures", action="store_true")
    p.add_argument("--include-replays", action="store_true")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=generate_dataset)
    a = ap.parse_args()
    a.run(a)


if __name__ == "__main__":
    main()
