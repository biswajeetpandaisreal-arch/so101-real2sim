"""Read-only hardware check: do the saved calibrations match the motors, and
what joint positions do the arms report? Nothing moves (torque is untouched).

    python scripts/check_arms.py --follower COM3 --leader COM4
"""

import argparse
from pathlib import Path

CALIB = Path(__file__).resolve().parent.parent / "config" / "calibration"


def check(name, device):
    try:
        device.bus.connect()
    except Exception as err:                       # unplugged, wrong port, no power
        print(f"{name}: could not connect ({err})")
        return
    try:
        print(f"{name}: calibration matches motors: {device.bus.is_calibrated}")
        pos = device.bus.sync_read("Present_Position")
        print("   " + "  ".join(f"{k}={v:7.1f}" for k, v in pos.items()))
    finally:
        device.bus.disconnect(disable_torque=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--follower", default="COM3")
    ap.add_argument("--leader", default="COM4")
    a = ap.parse_args()
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    check("follower", SO101Follower(SO101FollowerConfig(
        port=a.follower, id="so101_follower", calibration_dir=CALIB)))
    check("leader", SO101Leader(SO101LeaderConfig(
        port=a.leader, id="so101_leader", calibration_dir=CALIB)))


if __name__ == "__main__":
    main()
