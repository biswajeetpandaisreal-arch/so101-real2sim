"""Identify the simulated actuators from the real arm.

excite  The real follower moves on its own: from rest to a pose in the air, then
        one joint at a time through large steps, 1-2 deg steps (to expose any
        dead band) and a slow sine, then back to rest. Every sample logs the goal,
        the present position (degrees and raw encoder counts), load, current and
        time, with the servo registers that shape the response. Keep clear of the
        arm; Ctrl+C returns it to rest.
fit     For each joint, replay the logged goals in the twin and choose position
        gain kp, joint damping, Coulomb friction and armature by bounded least
        squares on that joint's tracking error. Then replay held-out
        teleoperation (leader commands as goals) with default and fitted
        parameters and compare with the follower's measured joints.

    python scripts/sysid.py excite
    python scripts/sysid.py fit --log data/sysid/excite_<time>.npz --validate data/so101_real
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from real2sim.data import load_episodes
from real2sim.hardware import follower
from real2sim.kinematics import min_jerk
from real2sim.sim import ACTUATORS, JOINTS, ROOT, SO101Env

HOME = np.array([0.0, -30.0, 20.0, 50.0, 0.0, 30.0])     # jaw tip ~15 cm above the table (checked in sim)
AMPLITUDE = np.array([15.0, 10.0, 10.0, 15.0, 30.0, 30.0])
REGISTERS = ["P_Coefficient", "I_Coefficient", "D_Coefficient", "CW_Dead_Zone", "CCW_Dead_Zone",
             "Minimum_Startup_Force", "Max_Torque_Limit", "Acceleration"]
PARAMS = ("kp", "damping", "frictionloss", "armature")
BOUNDS = np.array([(5.0, 5000.0), (1e-3, 20.0), (1e-5, 3.0), (1e-4, 0.5)])   # fitted in log space
PARKED = (0.8, 0.8)                                       # object out of reach during the fit


# ── excite ─────────────────────────────────────────────────────────────────────
def schedule():
    """(joint, label, duration s, offset from HOME in deg as a function of time)."""
    out = []
    for j, amp in enumerate(AMPLITUDE):
        out += [(j, f"step_large_{k}", 1.5, lambda t, o=o: o) for k, o in enumerate([amp, 0, -amp, 0])]
        out += [(j, f"step_small_{k}", 1.0, lambda t, o=o: o) for k, o in enumerate([2, 0, -2, 0, 1, 0, -1, 0])]
        out.append((j, "sine_0.5Hz", 4.0, lambda t, a=amp / 2: a * np.sin(2 * np.pi * 0.5 * t)))
    return out


def excite(_):
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


# ── fit ────────────────────────────────────────────────────────────────────────
def simulate(env, t, goal_deg, q0_deg):
    """Simulated joint positions at the log times t, each goal held until the next sample."""
    m, d = env.model, env.data
    env.reset(PARKED, 0.0, q0_lerobot=q0_deg)
    ctrl = env.lerobot_to_rad(goal_deg)
    out = np.empty_like(goal_deg)
    sim_t = 0.0
    for i, ti in enumerate(np.asarray(t) - t[0]):
        while sim_t + 1e-9 < ti:
            mujoco.mj_step(m, d)
            sim_t += m.opt.timestep
        out[i] = env.state_lerobot()
        d.ctrl[env.act] = ctrl[i]
    return out


def fit_joint(env, log, j):
    """(fitted params, RMSE with defaults, RMSE fitted) for joint j; env is left at its defaults."""
    idx = np.where(log["joint"] == j)[0]
    t, goal, pos = (log[k][idx[0]:idx[-1] + 1] for k in ("t", "goal", "pos"))
    name = JOINTS[j]
    default = env.get_actuators()[name]

    def residual(x):
        env.set_actuators({name: dict(zip(PARAMS, np.exp(x)))})
        return simulate(env, t, goal, pos[0])[:, j] - pos[:, j]

    lo, hi = np.log(BOUNDS[:, 0]), np.log(BOUNDS[:, 1])
    x0 = np.clip(np.log([default[k] for k in PARAMS]), lo + 1e-6, hi - 1e-6)
    rmse_default = float(np.sqrt(np.mean(residual(x0) ** 2)))
    r = least_squares(residual, x0, bounds=(lo, hi), diff_step=1e-3, max_nfev=200)
    env.set_actuators({name: default})
    return dict(zip(PARAMS, map(float, np.exp(r.x)))), rmse_default, float(np.sqrt(np.mean(r.fun ** 2)))


def steady_state_error(log):
    """Mean |goal - position| over the last 10% of each step hold, by joint and step size (deg)."""
    out = {}
    for j, name in enumerate(JOINTS):
        rows = {}
        for label in np.unique(log["label"][log["joint"] == j]):
            if not str(label).startswith("step"):
                continue
            idx = np.where((log["joint"] == j) & (log["label"] == label))[0]
            tail = idx[-max(3, len(idx) // 10):]
            size = abs(log["goal"][tail, j].mean() - log["goal"][idx[0] - 1, j])
            rows.setdefault(f"step {size:.0f}", []).append(np.abs(log["goal"][tail, j] - log["pos"][tail, j]).mean())
        out[name] = {k: round(float(np.mean(v)), 2) for k, v in sorted(rows.items())}
    return out


def teleop_rmse(env, episodes):
    """Per-joint RMSE between simulated and measured follower joints, driving the twin with the leader commands."""
    errors = []
    for ep in episodes:
        env.reset(PARKED, 0.0, q0_lerobot=ep["state"][0])
        sim = []
        for act in ep["action"]:
            sim.append(env.state_lerobot())
            env.step(act)
        errors.append(np.array(sim) - ep["state"])
    e = np.concatenate(errors)
    return {j: round(float(np.sqrt(np.mean(e[:, i] ** 2))), 2) for i, j in enumerate(JOINTS)}


def fit(a):
    z = np.load(a.log)
    log = {k: z[k] for k in ("t", "goal", "pos", "joint", "label")}
    env = SO101Env(actuators=None)
    fitted, table = {}, {}
    for j, name in enumerate(JOINTS):
        params, before, after = fit_joint(env, log, j)
        fitted[name] = params
        table[name] = {"rmse_default_deg": round(before, 2), "rmse_fitted_deg": round(after, 2),
                       "default": {k: round(env.get_actuators()[name][k], 4) for k in PARAMS},
                       "fitted": {k: round(v, 4) for k, v in params.items()}}
        print(f"{name:13s} excitation RMSE {before:5.2f} -> {after:5.2f} deg  "
              + "  ".join(f"{k}={v:.4g}" for k, v in params.items()), flush=True)
    report = {"excitation": table, "dead_band_deg": steady_state_error(log),
              "servo_registers": json.loads(str(z["registers"]))}
    if a.validate:
        episodes = load_episodes(Path(a.validate), a.repo_id)
        report["validation_rmse_deg"] = {"default": teleop_rmse(SO101Env(actuators=None), episodes),
                                         "fitted": teleop_rmse(SO101Env(actuators=fitted), episodes)}
        print("held-out teleoperation RMSE (deg):", json.dumps(report["validation_rmse_deg"]))
    Path(a.out).write_text(json.dumps(fitted, indent=1))
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1))
    print(f"wrote {a.out} and {a.report}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(required=True)
    sub.add_parser("excite", help="move the real follower and log its response").set_defaults(run=excite)
    p = sub.add_parser("fit", help="fit the twin's actuators to a log")
    p.add_argument("--log", required=True, help="data/sysid/excite_<time>.npz")
    p.add_argument("--validate", default=None, help="teleoperation dataset folder (held out)")
    p.add_argument("--repo-id", default="local/so101_real")
    p.add_argument("--out", default=str(ACTUATORS))
    p.add_argument("--report", default=str(ROOT / "results" / "sysid.json"))
    p.set_defaults(run=fit)
    a = ap.parse_args()
    a.run(a)


if __name__ == "__main__":
    main()
