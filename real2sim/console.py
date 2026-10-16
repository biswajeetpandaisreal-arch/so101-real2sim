"""Operator input while a control loop keeps running.

A background thread reads whole lines from stdin, so typed commands reach the
script in any terminal (including embedded ones, where per-key console APIs can
miss key presses), and LeRobot's per-step clamping warnings are silenced so the
prompts stay readable.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading


def quiet_lerobot():
    logging.getLogger().setLevel(logging.ERROR)


class LineReader:
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

    def wait(self, valid=None) -> str:
        while True:
            line = self._q.get()
            if valid is None or line in valid:
                return line
            print(f"  type one of: {', '.join(repr(v) for v in valid)}")
