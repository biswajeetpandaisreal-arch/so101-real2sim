"""Run generated trajectories open loop on the REAL arm.

For each test mark (never demonstrated), a trajectory is generated from the
nearest source demonstration the twin reproduces, checked in simulation, then
streamed to the real follower at 30 Hz with the real object on the mark. No
camera and no policy: this tests whether the generator's output is physically
valid on the real arm. The operator scores each run; results are saved after
every trial, with the follower's measured joints.

    python scripts/execute_real.py --marks 1,6 --trials 3

Typed + Enter: '' runs the trial, 's' skips it; after a run 'y' = in the box,
'1' = no contact, '2' = pushed, '3' = lifted but not delivered.
Keep clear of the arm; Ctrl+C returns it to rest.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from real2sim.data import load_demos, read_episode_list, replay
from real2sim.generate import generate
from real2sim.hardware import LineReader, follower, quiet_lerobot
from real2sim.kinematics import SO101IK, min_jerk
from real2sim.sim import FPS, JOINTS, ROOT, SO101Env, read_layout

OUTCOMES = {"y": "success", "1": "no_contact", "2": "pushed", "3": "not_delivered"}


def plan(env, ik, demos, target_xy, rank):
    """(commands, source episode) from the rank-th nearest reproducible demonstration
    whose generated trajectory succeeds in simulation, or (None, None)."""
    order = sorted(demos, key=lambda d: np.linalg.norm(d["info"]["xy"] - target_xy))
    for d in order[rank:] + order[:rank]:
        if not replay(env, d):
            continue
        commands = generate(env, ik, d["action"], d["info"]["xy"], d["info"]["yaw"], target_xy, 0.0)
        env.reset(target_xy, 0.0, q0_lerobot=commands[0])
        for c in commands:
            env.step(c)
        if env.success():
            return commands, d["episode"]
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/so101_real")
    ap.add_argument("--repo-id", default="local/so101_real")
    ap.add_argument("--episodes", default=str(ROOT / "results" / "episodes.csv"))
    ap.add_argument("--marks", default="1,6")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default="results/real_execution")
    a = ap.parse_args()
    layout = read_layout()
    demos = load_demos(Path(a.data), a.repo_id, read_episode_list(Path(a.episodes)))
    env = SO101Env()
    ik = SO101IK(env)
    plans = []
    for mark in a.marks.split(","):
        for k in range(a.trials):
            commands, source = plan(env, ik, demos, layout[mark]["xy"], k)
            plans.append({"mark": mark, "role": layout[mark]["role"], "trial": k, "source": source,
                          "commands": commands})
            status = "ok" if commands is not None else "FAILED"
            print(f"mark {mark} trial {k}: source episode {source}, {status} in sim")

    quiet_lerobot()
    robot = follower()
    robot.connect()
    keys = LineReader()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    results, logs = [], {}

    def position():
        p = robot.bus.sync_read("Present_Position")
        return np.array([p[j] for j in JOINTS])

    def stream(commands, log):
        for c in commands:
            t0 = time.perf_counter()
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, c)})
            log.append([time.perf_counter(), *c, *position()])
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))

    def glide(to, seconds, log):
        stream(min_jerk(position(), to, np.linspace(0, 1, int(seconds * FPS))[:, None]), log)

    def save():
        (out / f"results_{stamp}.json").write_text(json.dumps(results, indent=1))
        np.savez(out / f"joint_logs_{stamp}.npz", **logs,
                 columns=np.array(["t"] + [f"goal_{j}" for j in JOINTS] + [f"pos_{j}" for j in JOINTS]))

    rest = position()
    try:
        for p in plans:
            entry = {k: p[k] for k in ("mark", "role", "trial", "source")}
            if p["commands"] is None:
                results.append(entry | {"outcome": "no_sim_plan"})
                continue
            print(f"\nPlace the object on mark {p['mark']} ({p['role']}). Enter = run, s + Enter = skip", flush=True)
            if keys.wait(("", "s")) == "s":
                continue
            log = []
            glide(p["commands"][0], 3.0, log)
            stream(p["commands"], log)
            glide(rest, 3.0, log)
            print("Outcome + Enter: y = in the box, 1 = no contact, 2 = pushed, 3 = lifted not delivered", flush=True)
            outcome = OUTCOMES[keys.wait(tuple(OUTCOMES))]
            results.append(entry | {"outcome": outcome})
            logs[f"{p['mark']}_{p['trial']}"] = np.array(log)
            save()
    except KeyboardInterrupt:
        print("interrupted: returning to rest")
        glide(rest, 3.0, [])
    finally:
        robot.disconnect()
        save()
    for role in ("interp", "extrap"):
        r = [x for x in results if x["role"] == role and x["outcome"] != "no_sim_plan"]
        print(f"{role}: {sum(x['outcome'] == 'success' for x in r)}/{len(r)} succeeded on the real arm")


if __name__ == "__main__":
    main()
