"""Identify the simulated actuators from the real arm's measured response.

Fit:      for each joint, replay the goals logged in its excitation segment
          (scripts/sysid_excite.py) and choose position gain kp, joint damping,
          Coulomb friction (frictionloss) and armature by bounded least squares
          on that joint's tracking error.
Validate: replay held-out teleoperation recordings (leader commands as goals)
          with the default and the fitted parameters; compare the simulated
          joint trajectories with the follower's measured ones.
Explain:  steady-state error at the end of every step, next to the servo
          registers that set it.

    python scripts/sysid_fit.py --excite data/sysid/excite_<time>.npz --validate data/so101_real
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from real2sim.dataset_io import load_episodes
from real2sim.sim_env import ACTUATORS, JOINTS, SO101Env

PARAMS = ("kp", "damping", "frictionloss", "armature")
BOUNDS = np.array([(5.0, 5000.0), (1e-3, 20.0), (1e-5, 3.0), (1e-4, 0.5)])   # fitted in log space
PARKED = (0.8, 0.8)                                   # object out of reach during the fit


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--excite", required=True)
    ap.add_argument("--validate", default=None, help="teleoperation dataset folder (held out)")
    ap.add_argument("--repo-id", default="local/so101_real")
    ap.add_argument("--out", default=str(ACTUATORS))
    ap.add_argument("--report", default="results/sysid.json")
    a = ap.parse_args()
    z = np.load(a.excite)
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


if __name__ == "__main__":
    main()
