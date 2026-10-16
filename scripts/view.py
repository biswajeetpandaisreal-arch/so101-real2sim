"""Watch the twin in MuJoCo's viewer: a recorded demonstration replayed, then
the same demonstration adapted by the generator to other marks.

Works from a fresh clone (uses results/sample_demo.npz, one recorded
demonstration's joint commands). Close the window to stop. On macOS run it with
`mjpython` instead of `python`.

    python scripts/view.py                 # replay, then marks 1 and 6
    python scripts/view.py --marks 1 6 2   # any marks from config/layout.csv
"""

import argparse
import time

import numpy as np
from mujoco import viewer

from real2sim.generate import generate
from real2sim.kinematics import SO101IK
from real2sim.sim import FPS, ROOT, SO101Env, read_layout


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--marks", nargs="+", default=["1", "6"])
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed")
    a = ap.parse_args()
    demo = np.load(ROOT / "results" / "sample_demo.npz")
    xy, yaw, actions = demo["xy"], float(demo["yaw"]), demo["action"].astype(float)
    env = SO101Env()
    ik = SO101IK(env)
    layout = read_layout()
    clips = [(f"recorded demonstration (episode {int(demo['episode'])}, mark {demo['mark']})", xy, yaw, actions)]
    for mark in a.marks:
        print(f"generating a trajectory for mark {mark} ...", flush=True)
        clips.append((f"generated for mark {mark} ({layout[mark]['role']})", layout[mark]["xy"], 0.0,
                      generate(env, ik, actions, xy, yaw, layout[mark]["xy"], 0.0)))
    with viewer.launch_passive(env.model, env.data) as v:
        while v.is_running():
            for name, obj_xy, obj_yaw, commands in clips:
                env.reset(obj_xy, obj_yaw, q0_lerobot=commands[0])
                for c in commands:
                    if not v.is_running():
                        return
                    t0 = time.perf_counter()
                    env.step(c)
                    v.sync()
                    time.sleep(max(0.0, 1.0 / (FPS * a.speed) - (time.perf_counter() - t0)))
                print(f"{name}: {'in the box' if env.success() else 'missed'}", flush=True)
                time.sleep(1.0)


if __name__ == "__main__":
    main()
