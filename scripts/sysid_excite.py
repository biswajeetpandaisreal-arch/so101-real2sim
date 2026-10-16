"""Excite each follower joint and log the raw servo response, for identifying
the simulated actuators (scripts/sysid_fit.py).

The arm moves on its own: from rest to a pose in the air, then one joint at a
time through large steps, 1-2 deg steps (to expose any dead band) and a slow
sine, then back to rest. Every sample logs the commanded goal, the present
position (degrees and raw encoder counts), load, current and time, alongside
the servo registers that shape the response (PID gains, dead zones, minimum
start-up force).

Keep clear of the arm; Ctrl+C returns it to rest.

    python scripts/sysid_excite.py
"""

import json
import time

import numpy as np

from real2sim.hardware import follower
from real2sim.kinematics import min_jerk
from real2sim.sim_env import JOINTS, ROOT

HOME = np.array([0.0, -30.0, 20.0, 50.0, 0.0, 30.0])     # jaw tip ~15 cm above the table (checked in sim)
AMPLITUDE = np.array([15.0, 10.0, 10.0, 15.0, 30.0, 30.0])
REGISTERS = ["P_Coefficient", "I_Coefficient", "D_Coefficient", "CW_Dead_Zone", "CCW_Dead_Zone",
             "Minimum_Startup_Force", "Max_Torque_Limit", "Acceleration"]


def schedule():
    """(joint, label, duration s, offset from HOME in deg as a function of time)."""
    out = []
    for j, amp in enumerate(AMPLITUDE):
        out += [(j, f"step_large_{k}", 1.5, lambda t, o=o: o) for k, o in enumerate([amp, 0, -amp, 0])]
        out += [(j, f"step_small_{k}", 1.0, lambda t, o=o: o) for k, o in enumerate([2, 0, -2, 0, 1, 0, -1, 0])]
        out.append((j, "sine_0.5Hz", 4.0, lambda t, a=amp / 2: a * np.sin(2 * np.pi * 0.5 * t)))
    return out


def main():
    robot = follower(max_step_deg=None)                  # steps are the point; no command clamping
    robot.connect()
    bus = robot.bus
    registers = {r: {m: int(bus.read(r, m, normalize=False)) for m in JOINTS} for r in REGISTERS}
    log = {k: [] for k in ("t", "goal", "pos", "raw", "load", "current", "joint", "label")}

    def read_position():
        p = bus.sync_read("Present_Position")
        return np.array([p[m] for m in JOINTS])

    def command(goal, joint, label):
        bus.sync_write("Goal_Position", {m: float(g) for m, g in zip(JOINTS, goal)})
        raw, load, current = (bus.sync_read(r, normalize=False)
                              for r in ("Present_Position", "Present_Load", "Present_Current"))
        log["t"].append(time.perf_counter())
        log["goal"].append(list(goal))
        log["pos"].append(list(read_position()))
        log["raw"].append([raw[m] for m in JOINTS])
        log["load"].append([load[m] for m in JOINTS])
        log["current"].append([current[m] for m in JOINTS])
        log["joint"].append(joint)
        log["label"].append(label)

    def move(a, b, seconds, label):
        t0 = time.perf_counter()
        while (s := (time.perf_counter() - t0) / seconds) < 1.0:
            command(min_jerk(a, b, s), -1, label)

    rest = read_position()
    try:
        move(rest, HOME, 4.0, "to_home")
        for j, label, duration, offset in schedule():
            print(f"{JOINTS[j]:13s} {label}", flush=True)
            t0 = time.perf_counter()
            while (t := time.perf_counter() - t0) < duration:
                goal = HOME.copy()
                goal[j] += offset(t)
                command(goal, j, label)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        move(read_position(), rest, 4.0, "to_rest")
        robot.disconnect()
    out = ROOT / "data" / "sysid" / f"excite_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.array(v) for k, v in log.items()}, joints=np.array(JOINTS),
             registers=json.dumps(registers), home=HOME)
    print(f"saved {out}: {len(log['t'])} samples at {len(log['t']) / (log['t'][-1] - log['t'][0]):.0f} Hz")


if __name__ == "__main__":
    main()
