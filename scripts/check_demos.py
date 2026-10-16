"""Quick report on recorded demonstrations: for each episode, whether the jaws
closed on the object, where the object was (from the grasp), how long the
operator was active, and whether the episode succeeds when replayed in sim with
the object at that position.

    python scripts/check_demos.py --data data/so101_real
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.dataset_io import load_episodes  # noqa: E402
from real2sim.demo_analysis import active_span, executed, object_position, task_span  # noqa: E402
from real2sim.kinematics import SO101IK  # noqa: E402
from real2sim.layout import PAN_AXIS_X  # noqa: E402
from real2sim.sim_env import FPS, SO101Env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/so101_real")
    ap.add_argument("--repo-id", default="local/so101_real")
    a = ap.parse_args()
    env = SO101Env()
    ik = SO101IK(env)
    for ep in load_episodes(Path(a.data), a.repo_id):
        A, S = ep["action"], ep["state"]
        span = active_span(S)
        xy, k = object_position(env, ik, A, S)
        head = f"ep {ep['episode']:2d} ({len(S) / FPS:4.1f}s)"
        if span is None:
            print(f"{head}: idle the whole time -> unusable")
            continue
        if xy is None:
            print(f"{head}: active {span[0] / FPS:.0f}-{span[1] / FPS:.0f}s, jaws never closed on the object -> unusable")
            continue
        a0, a1 = task_span(A, S)
        env.reset(xy, 0.0, q0_lerobot=S[a0])
        lift = 0.0
        for act in executed(A, S)[a0:a1]:
            env.step(act)
            lift = max(lift, env.cube_pos()[2])
        end = (env.cube_pos()[:2] - [PAN_AXIS_X, 0]) * 100
        released = bool((A[k:a1, 5] > 30).any())
        print(f"{head}: task {a0 / FPS:.0f}-{a1 / FPS:.0f}s, object at {100 * (xy[0] - PAN_AXIS_X):.1f} fwd "
              f"{100 * xy[1]:.1f} left, gripper opened after grasp: {released}; sim replay "
              f"{'SUCCESS' if env.success() else 'fail'} (lifted {100 * lift:.0f} cm, ends {end[0]:.0f} fwd {end[1]:.0f} left)")


if __name__ == "__main__":
    main()
