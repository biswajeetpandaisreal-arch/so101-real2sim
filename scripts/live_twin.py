"""Live digital twin: the leader arm drives the real follower and the MuJoCo
SO-101 at the same time, so the two can be compared side by side.

Windows lets only one program open a COM port, so this replaces
`lerobot-teleoperate` while it runs. Close the viewer window or press Ctrl+C
to stop; the follower's torque is then released (support the arm).

    python scripts/live_twin.py --follower COM3 --leader COM4
    python scripts/live_twin.py --no-follower          # sim only, leader drives it
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from mujoco import viewer as mj_viewer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.sim_env import FPS, JOINTS, SO101Env  # noqa: E402

CALIB = Path(__file__).resolve().parent.parent / "config" / "calibration"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--follower", default="COM3")
    ap.add_argument("--leader", default="COM4")
    ap.add_argument("--no-follower", action="store_true", help="drive only the simulation")
    ap.add_argument("--max-step", type=float, default=5.0,
                    help="largest follower joint change per command (deg)")
    ap.add_argument("--cube", type=float, nargs=2, default=(0.24, 0.02), help="sim cube x y (m)")
    a = ap.parse_args()

    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    leader = SO101Leader(SO101LeaderConfig(port=a.leader, id="so101_leader", calibration_dir=CALIB))
    leader.connect()
    robot = None
    if not a.no_follower:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        robot = SO101Follower(SO101FollowerConfig(
            port=a.follower, id="so101_follower", calibration_dir=CALIB,
            max_relative_target=a.max_step))
        robot.connect()

    def read():
        act = leader.get_action()
        return act, np.array([act[f"{j}.pos"] for j in JOINTS])

    env = SO101Env()
    _, q0 = read()
    env.reset(a.cube, 0.0, q0_lerobot=q0)
    print("Running. Close the viewer window or press Ctrl+C to stop.")
    try:
        with mj_viewer.launch_passive(env.model, env.data) as v:
            while v.is_running():
                t0 = time.perf_counter()
                act, vec = read()
                if robot is not None:
                    robot.send_action(act)
                env.step(vec)
                v.sync()
                time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if robot is not None:
            robot.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
