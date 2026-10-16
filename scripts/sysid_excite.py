"""Excite each follower joint and log the raw servo response, for system
identification of the simulated actuators.

The arm moves on its own: from its resting pose to a home pose in the air, then
one joint at a time through large steps, small steps (to expose the dead band)
and a slow sine, then back to rest. Every loop logs the commanded goal, the
present position (degrees and raw encoder counts), load, current and a
timestamp, plus the servo control registers (P/I/D gains, dead zones,
minimum start-up force) that shape the response.

Keep clear of the arm; Ctrl+C returns it to rest.

    python scripts/sysid_excite.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.sim_env import JOINTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "config" / "calibration"
HOME = np.array([0.0, -30.0, 20.0, 50.0, 0.0, 30.0])     # tip ~15 cm above the table (checked in sim)
AMP = np.array([15.0, 10.0, 10.0, 15.0, 30.0, 30.0])     # large-step amplitude per joint
REGISTERS = ["P_Coefficient", "I_Coefficient", "D_Coefficient", "CW_Dead_Zone", "CCW_Dead_Zone",
             "Minimum_Startup_Force", "Max_Torque_Limit", "Acceleration"]


def schedule():
    """List of (joint, label, duration_s, target_fn(t) -> offset from home, deg)."""
    out = []
    for j in range(6):
        A = AMP[j]
        for k, off in enumerate([A, 0, -A, 0]):
            out.append((j, f"step_large_{k}", 1.5, lambda t, o=off: o))
        for k, off in enumerate([2, 0, -2, 0, 1, 0, -1, 0]):
            out.append((j, f"step_small_{k}", 1.0, lambda t, o=off: o))
        out.append((j, "sine_0.5Hz", 4.0, lambda t, a=A / 2: a * np.sin(2 * np.pi * 0.5 * t)))
    return out


def min_jerk(a, b, s):
    s = np.clip(s, 0, 1)
    return a + (b - a) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def main():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(port="COM3", id="so101_follower", calibration_dir=CALIB))
    robot.connect()
    bus = robot.bus
    regs = {r: {m: int(bus.read(r, m, normalize=False)) for m in JOINTS} for r in REGISTERS}
    print(json.dumps(regs))
    log = {k: [] for k in ("t", "goal", "pos", "raw", "load", "current", "joint", "label")}

    def send(goal):
        bus.sync_write("Goal_Position", {m: float(g) for m, g in zip(JOINTS, goal)})

    def sample(goal, j, label):
        pos = bus.sync_read("Present_Position")
        raw = bus.sync_read("Present_Position", normalize=False)
        load = bus.sync_read("Present_Load", normalize=False)
        cur = bus.sync_read("Present_Current", normalize=False)
        log["t"].append(time.perf_counter())
        log["goal"].append(list(goal))
        log["pos"].append([pos[m] for m in JOINTS])
        log["raw"].append([int(raw[m]) for m in JOINTS])
        log["load"].append([int(load[m]) for m in JOINTS])
        log["current"].append([int(cur[m]) for m in JOINTS])
        log["joint"].append(j)
        log["label"].append(label)

    def move(a, b, seconds, label):
        t0 = time.perf_counter()
        while (s := (time.perf_counter() - t0) / seconds) < 1.0:
            g = min_jerk(a, b, s)
            send(g)
            sample(g, -1, label)

    start = np.array([bus.sync_read("Present_Position")[m] for m in JOINTS])
    print("start pose", start.round(1))
    try:
        move(start, HOME, 4.0, "to_home")
        for j, label, dur, fn in schedule():
            print(f"{JOINTS[j]:13s} {label}")
            t0 = time.perf_counter()
            while (t := time.perf_counter() - t0) < dur:
                g = HOME.copy()
                g[j] += fn(t)
                send(g)
                sample(g, j, label)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        here = np.array([bus.sync_read("Present_Position")[m] for m in JOINTS])
        move(here, start, 4.0, "to_rest")
        time.sleep(0.5)
        robot.disconnect()
    out = ROOT / "data" / "sysid" / f"excite_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.array(v) for k, v in log.items()}, joints=np.array(JOINTS),
             registers=json.dumps(regs), home=HOME)
    n = len(log["t"])
    print(f"saved {out}: {n} samples, {n / (log['t'][-1] - log['t'][0]):.0f} Hz")


if __name__ == "__main__":
    main()
