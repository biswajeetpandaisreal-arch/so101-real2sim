"""Live, readable status of a background lerobot recording (follows its log).

    python scripts/watch_log.py outputs/record_mark5.log
"""

import sys
import time
from pathlib import Path

SHOW = {"Recording episode": ">>> RECORDING episode {n}: do the demo, then press RIGHT ARROW",
        "Reset the environment": "--- PAUSE: put the tealight back, then press RIGHT ARROW to start the next demo",
        "Re-record episode": "<<< redoing that episode",
        "Stop recording": "=== recording stopped",
        "utils.py:147 Exiting": "=== recorder closed (batch finished)",
        "arrow key pressed": "    (key received)"}


def main():
    path = Path(sys.argv[1])
    pos = 0
    while True:
        text = path.read_bytes().decode("utf-8", "replace") if path.exists() else ""
        for line in text[pos:].splitlines():
            for key, msg in SHOW.items():
                if key in line:
                    n = line.rsplit(" ", 1)[-1] if key == "Recording episode" else ""
                    stamp = line[5:24] if line.startswith("INFO") else " " * 19
                    print(f"{stamp}  {msg.format(n=n)}", flush=True)
        pos = len(text)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
