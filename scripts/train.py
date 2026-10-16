"""`lerobot-train` with a Windows fix: takes the same arguments.

LeRobot marks the newest checkpoint with a `checkpoints/last` symlink, which
Windows refuses to create without Developer Mode, crashing training at its
first checkpoint. Here the newest checkpoint is recorded in
`checkpoints/last.txt` instead.

    python scripts/train.py --policy.type=act --dataset.root=data/train ...
"""

import sys
from pathlib import Path

import lerobot.scripts.lerobot_train as lt


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    marker = checkpoint_dir.parent / "last.txt"
    marker.write_text(checkpoint_dir.name)
    return marker


if sys.platform == "win32":
    lt.update_last_checkpoint = update_last_checkpoint

if __name__ == "__main__":
    lt.main()
