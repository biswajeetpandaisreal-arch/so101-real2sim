"""Record real SO-101 demonstrations with LeRobot's recorder.

Each frame stores the leader's joint command, the follower's joint positions
(30 Hz) and two cameras: the follower's wrist camera and a side view.

    python scripts/record_real.py --episodes 6            # new dataset
    python scripts/record_real.py --episodes 6 --resume   # add episodes

Keys work in any window. Right arrow ends a demonstration; the pause that
follows waits for the operator (up to --reset-time s, the follower still
follows the leader) and Right arrow starts the next one. Left arrow redoes the
current one; Esc during a pause stops.

LeRobot's key listener is global and also maps the letters n / r / q to
next / re-record / quit, so typing anywhere on the computer ended or restarted
demonstrations. Only the arrows and Esc are listened to here.
"""

import argparse
import sys

import lerobot.scripts.lerobot_record as lerobot_record
from lerobot.utils.keyboard_input import apply_recording_control, create_key_listener

from real2sim.data import TASK
from real2sim.hardware import CALIBRATION_DIR, FOLLOWER_PORT, LEADER_PORT

CAMERAS = ("{ wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, "
           "side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }")


def arrows_only_listener():
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    def on_key(name: str) -> None:
        if name in ("right", "left", "esc"):
            apply_recording_control(name, events)

    return create_key_listener(on_key, controls_help="Right / Left / Esc"), events


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--root", default="data/so101_real")
    ap.add_argument("--episode-time", type=int, default=60)
    ap.add_argument("--reset-time", type=int, default=600)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")                  # LeRobot prints non-ASCII characters
    lerobot_record.init_keyboard_listener = arrows_only_listener
    sys.argv = ["lerobot-record",
                "--robot.type=so101_follower", f"--robot.port={FOLLOWER_PORT}", "--robot.id=so101_follower",
                f"--robot.calibration_dir={CALIBRATION_DIR}", "--robot.max_relative_target=10.0",
                f"--robot.cameras={CAMERAS}",
                "--teleop.type=so101_leader", f"--teleop.port={LEADER_PORT}", "--teleop.id=so101_leader",
                f"--teleop.calibration_dir={CALIBRATION_DIR}",
                "--dataset.repo_id=local/so101_real", f"--dataset.root={a.root}", f"--dataset.single_task={TASK}",
                f"--dataset.num_episodes={a.episodes}", "--dataset.fps=30",
                f"--dataset.episode_time_s={a.episode_time}", f"--dataset.reset_time_s={a.reset_time}",
                "--dataset.push_to_hub=false", "--play_sounds=false", f"--resume={str(a.resume).lower()}"]
    lerobot_record.main()


if __name__ == "__main__":
    main()
