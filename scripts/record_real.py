"""Record real SO-101 demonstrations with lerobot-record: joint angles (leader
command + follower position, 30 Hz) and two cameras (wrist = USB camera,
side = laptop webcam).

    python scripts/record_real.py --episodes 6            # first batch (new dataset)
    python scripts/record_real.py --episodes 6 --resume   # add a batch to it

Keys (work in any window): Right arrow ends the demo; the pause after it waits
for you (up to --reset-time s, the arm still follows the leader) and Right
arrow starts the next demo. Left arrow = redo, Esc (during a pause) = stop.

LeRobot's listener is global and, besides the arrows, treats the letters n / r / q
as next / re-record / quit, so typing anywhere on the computer (a chat message
containing "n") ended or restarted demos. Here only the arrows and Esc count.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAMERAS = ("{ wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, "
           "side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }")


def arrows_only_listener():
    from lerobot.utils.keyboard_input import apply_recording_control, create_key_listener
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    def on_key(name: str) -> None:
        if name in ("right", "left", "esc"):
            apply_recording_control(name, events)

    return create_key_listener(on_key, controls_help="Right / Left / Esc only"), events


def run_recorder(argv):
    import lerobot.scripts.lerobot_record as rec
    rec.init_keyboard_listener = arrows_only_listener
    sys.argv = ["lerobot-record", *argv]
    rec.main()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--recorder":   # child process: the patched recorder
        return run_recorder(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--episode-time", type=int, default=60)
    ap.add_argument("--reset-time", type=int, default=600)
    a = ap.parse_args()
    cmd = [sys.executable, __file__, "--recorder",
           "--robot.type=so101_follower", "--robot.port=COM3", "--robot.id=so101_follower",
           "--robot.calibration_dir=config/calibration", "--robot.max_relative_target=10.0",
           f"--robot.cameras={CAMERAS}",
           "--teleop.type=so101_leader", "--teleop.port=COM4", "--teleop.id=so101_leader",
           "--teleop.calibration_dir=config/calibration",
           "--dataset.repo_id=local/so101_real", "--dataset.root=data/so101_real",
           "--dataset.single_task=Pick up the red cube and place it in the blue basket.",
           f"--dataset.num_episodes={a.episodes}", "--dataset.fps=30",
           f"--dataset.episode_time_s={a.episode_time}", f"--dataset.reset_time_s={a.reset_time}",
           "--dataset.push_to_hub=false", "--play_sounds=false",
           f"--resume={str(a.resume).lower()}"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")       # lerobot prints non-ASCII characters
    sys.exit(subprocess.call(cmd, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
