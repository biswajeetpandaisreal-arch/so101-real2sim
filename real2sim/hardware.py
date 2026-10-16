"""The real SO-101 follower, the leader's port and operator input.

Ports and calibration live here so every hardware script agrees on them; the
calibration files are the ones LeRobot wrote when the arms were calibrated.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from pathlib import Path

CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "config" / "calibration"
FOLLOWER_PORT, LEADER_PORT = "COM3", "COM4"


def follower(port: str = FOLLOWER_PORT, max_step_deg: float | None = 10.0):
    """SO-101 follower. max_step_deg caps the change of any joint target per
    command (LeRobot's max_relative_target) so a bad command cannot slam the arm."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    return SO101Follower(SO101FollowerConfig(port=port, id="so101_follower", calibration_dir=CALIBRATION_DIR,
                                             max_relative_target=max_step_deg))


def quiet_lerobot():
    """Silence LeRobot's per-command clamping warnings so prompts stay readable."""
    logging.getLogger().setLevel(logging.ERROR)


class LineReader:
    """Operator commands typed as lines while a control loop keeps running.

    A background thread reads stdin, so input works in any terminal (per-key
    console APIs miss key presses in embedded terminals)."""

    def __init__(self):
        self._q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        for line in sys.stdin:
            self._q.put(line.strip().lower())

    def poll(self) -> str | None:
        """The next typed line ('' for a bare Enter), or None if nothing yet."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def wait(self, valid: tuple[str, ...]) -> str:
        while (line := self._q.get()) not in valid:
            print(f"  type one of: {', '.join(repr(v) for v in valid)}")
        return line
