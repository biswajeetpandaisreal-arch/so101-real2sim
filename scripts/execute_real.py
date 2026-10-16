"""Open-loop execution of generated trajectories on the REAL arm.

For each test spot the operator never demonstrated (config/layout.csv, roles
interp / extrap), a trajectory is generated from the nearest recorded
demonstration, checked in simulation, then streamed to the real follower at
30 Hz with the real cube on the taped cross. No camera, no policy: this tests
whether the generator's output is physically valid in the real world.

The operator scores each run by key (y = cube in basket; otherwise the failure
type). The follower's measured joint positions are logged during every run.

    python scripts/execute_real.py --data data/so101_real --trials 2
Type then Enter: '' = run, s = skip; after a run: y success, 1 no contact, 2 pushed,
3 lifted but not delivered.  Keep clear of the arm; Ctrl+C returns it to rest.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.console import LineReader, quiet_lerobot  # noqa: E402
from real2sim.dataset_io import load_episodes  # noqa: E402
from real2sim.generate import generate  # noqa: E402
from real2sim.kinematics import SO101IK  # noqa: E402
from real2sim.layout import read_layout  # noqa: E402
from real2sim.sim_env import FPS, JOINTS, SO101Env  # noqa: E402
from scripts.replay import load_demos, read_spots, replay  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "config" / "calibration"
OUTCOMES = {"y": "success", "1": "no_contact", "2": "pushed", "3": "not_delivered"}


def plan_for(env, ik, eps, spots, target_xy, rank):
    """Generated trajectory for target_xy from the rank-th nearest source whose
    replay succeeds; falls through to further sources until one succeeds in sim."""
    order = sorted(eps, key=lambda e: np.linalg.norm(np.asarray(spots[e["episode"]]["xy"]) - target_xy))
    for ep in order[rank:] + order[:rank]:
        sp = spots[ep["episode"]]
        if not replay(env, ep, sp)[0]:
            continue
        acts = generate(env, ik, ep["action"], sp["xy"], sp["yaw"], target_xy, 0.0)
        env.reset(target_xy, 0.0, q0_lerobot=acts[0])
        for a in acts:
            env.step(a)
        if env.success():
            return acts, ep["episode"]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/so101_real")
    ap.add_argument("--repo-id", default="local/so101_real")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--spots", default="1,6")
    ap.add_argument("--out", default="results/real_execution")
    a = ap.parse_args()
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    layout = read_layout()
    spots = read_spots(Path(a.data) / "episodes.csv")
    eps = load_demos(Path(a.data), a.repo_id, spots)
    env = SO101Env()
    ik = SO101IK(env)
    plans = []
    for name in a.spots.split(","):
        for k in range(a.trials):
            acts, src = plan_for(env, ik, eps, spots, layout[name]["xy"], k)
            plans.append({"spot": name, "role": layout[name]["role"], "trial": k, "source": src, "acts": acts})
            print(f"{name} trial {k}: source episode {src}, sim {'ok' if acts is not None else 'FAILED'}")
    robot = SO101Follower(SO101FollowerConfig(port="COM3", id="so101_follower",
                                              calibration_dir=CALIB, max_relative_target=10.0))
    quiet_lerobot()
    robot.connect()
    keys = LineReader()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    results, logs = [], {}

    def read():
        p = robot.bus.sync_read("Present_Position")
        return np.array([p[j] for j in JOINTS])

    def stream(traj, log):
        for g in traj:
            t0 = time.perf_counter()
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, g)})
            log.append([time.perf_counter(), *g, *read()])
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))

    def glide(a_, b_, seconds, log):
        s = np.linspace(0, 1, int(seconds * FPS))[:, None]
        stream(a_ + (b_ - a_) * (10 * s**3 - 15 * s**4 + 6 * s**5), log)

    rest = read()
    try:
        for p in plans:
            if p["acts"] is None:
                results.append({k: p[k] for k in ("spot", "role", "trial", "source")} | {"outcome": "no_sim_plan"})
                continue
            print(f"\nPlace the cube on {p['spot']} ({p['role']}), square to the robot. "
                  f"Enter = run, s + Enter = skip", flush=True)
            if keys.wait(("", "s")) == "s":
                continue
            log = []
            glide(read(), p["acts"][0], 3.0, log)
            stream(p["acts"], log)
            glide(read(), rest, 3.0, log)
            print("Outcome + Enter:  y = in basket, 1 = no contact, 2 = pushed, "
                  "3 = lifted not delivered", flush=True)
            o = OUTCOMES[keys.wait(tuple(OUTCOMES))]
            print(f"  {o}")
            results.append({k: p[k] for k in ("spot", "role", "trial", "source")} | {"outcome": o})
            logs[f"{p['spot']}_{p['trial']}"] = np.array(log)
    except KeyboardInterrupt:
        print("interrupted: returning to rest")
        glide(read(), rest, 3.0, [])
    finally:
        robot.disconnect()
        (out / f"results_{stamp}.json").write_text(json.dumps(results, indent=1))
        np.savez(out / f"joint_logs_{stamp}.npz", **logs,
                 columns=np.array(["t"] + [f"goal_{j}" for j in JOINTS] + [f"pos_{j}" for j in JOINTS]))
    for role in ("interp", "extrap"):
        r = [x for x in results if x["role"] == role and x["outcome"] != "no_sim_plan"]
        print(f"{role}: {sum(x['outcome'] == 'success' for x in r)}/{len(r)} succeeded on the real arm")


if __name__ == "__main__":
    main()
