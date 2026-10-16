"""Identify the simulated actuators from the real arm's measured response.

Fit:      for each joint, replay the logged goal positions of its excitation
          segment (scripts/sysid_excite.py) in MuJoCo and choose position gain
          kp, joint damping, Coulomb friction (frictionloss) and armature by
          bounded least squares on the tracking error of that joint.
Validate: replay held-out teleoperation recordings (leader commands as goals)
          with the default and the fitted parameters and compare the simulated
          joint trajectories to the follower's measured ones.
Explain:  steady-state error after each step (the dead band) next to the
          servo registers that set it.

    python scripts/sysid_fit.py --excite data/sysid/excite_<time>.npz --validate data/so101_real
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real2sim.sim_env import ACTUATORS, JOINTS, SO101Env  # noqa: E402

PARAMS = ("kp", "damping", "frictionloss", "armature")
BOUNDS = {"kp": (5.0, 5000.0), "damping": (1e-3, 20.0), "frictionloss": (1e-5, 3.0),
          "armature": (1e-4, 0.5)}
FAR = (0.8, 0.8)                                  # cube parked out of reach during the fit


def simulate(env, t, goal_deg, q0_deg):
    """Joint positions (deg-like LeRobot units) at the log times t, with the goal
    held (zero-order) from each sample to the next."""
    m, d = env.model, env.data
    env.reset(FAR, 0.0, q0_lerobot=q0_deg)
    d.qvel[:] = 0
    t = np.asarray(t) - t[0]
    out = np.empty_like(goal_deg)
    sim_t, dt = 0.0, m.opt.timestep
    ctrl = env.lerobot_to_rad(goal_deg)
    for i in range(len(t)):
        while sim_t + 1e-9 < t[i]:
            mujoco.mj_step(m, d)
            sim_t += dt
        out[i] = env.state_lerobot()
        d.ctrl[env.act] = ctrl[i]
    return out


def segment(log, j):
    idx = np.where(log["joint"] == j)[0]
    return idx[0], idx[-1] + 1


def fit_joint(env, log, j, start):
    a, b = segment(log, j)
    t, goal, pos = log["t"][a:b], log["goal"][a:b], log["pos"][a:b]
    base = {k: start[JOINTS[j]][k] for k in PARAMS}

    def residual(x):
        env.set_actuators({JOINTS[j]: dict(zip(PARAMS, np.exp(x)))})
        return simulate(env, t, goal, pos[0])[:, j] - pos[:, j]

    lo = np.log([BOUNDS[k][0] for k in PARAMS])
    hi = np.log([BOUNDS[k][1] for k in PARAMS])
    x0 = np.clip(np.log([base[k] for k in PARAMS]), lo + 1e-6, hi - 1e-6)
    r = least_squares(residual, x0, bounds=(lo, hi), x_scale=1.0, diff_step=1e-3, max_nfev=200)
    fitted = dict(zip(PARAMS, map(float, np.exp(r.x))))
    env.set_actuators({JOINTS[j]: base})                       # leave env at defaults for others
    rmse0 = float(np.sqrt(np.mean(residual(x0) ** 2)))
    env.set_actuators({JOINTS[j]: base})
    return fitted, rmse0, float(np.sqrt(np.mean(r.fun ** 2)))


def dead_band(log):
    """Steady-state |goal - position| at the end of every hold, per joint and step size."""
    out = {}
    for j, name in enumerate(JOINTS):
        rows = {}
        for lab in np.unique(log["label"][log["joint"] == j]):
            if not str(lab).startswith("step"):
                continue
            idx = np.where((log["joint"] == j) & (log["label"] == lab))[0]
            tail = idx[-max(3, len(idx) // 10):]                 # last 10% of the hold
            g = log["goal"][tail, j].mean()
            amp = abs(g - log["goal"][idx[0] - 1, j]) if idx[0] > 0 else abs(g)
            err = float(np.abs(log["goal"][tail, j] - log["pos"][tail, j]).mean())
            rows.setdefault(f"step {amp:.0f}", []).append(err)
        out[name] = {k: round(float(np.mean(v)), 2) for k, v in sorted(rows.items())}
    return out


def validate(env_default, env_fit, data_root, repo_id):
    from real2sim.dataset_io import load_episodes
    eps = load_episodes(Path(data_root), repo_id)
    res = {}
    for name, env in (("default", env_default), ("fitted", env_fit)):
        errs = []
        for ep in eps:
            env.reset(FAR, 0.0, q0_lerobot=ep["state"][0])
            sim = []
            for act in ep["action"]:
                sim.append(env.state_lerobot())
                env.step(act)
            errs.append(np.array(sim) - ep["state"])
        e = np.concatenate(errs)
        res[name] = {j: round(float(np.sqrt(np.mean(e[:, i] ** 2))), 2) for i, j in enumerate(JOINTS)}
    return res


def load_log(path):
    z = np.load(path, allow_pickle=False)
    log = {k: z[k] for k in ("t", "goal", "pos", "raw", "load", "current", "joint", "label")}
    return log, json.loads(str(z["registers"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excite", required=True)
    ap.add_argument("--validate", default=None, help="teleop dataset folder (held out)")
    ap.add_argument("--repo-id", default="local/so101_real")
    ap.add_argument("--joints", default="0,1,2,3,4,5")
    ap.add_argument("--out", default=str(ACTUATORS))
    ap.add_argument("--report", default="results/sysid.json")
    a = ap.parse_args()
    log, regs = load_log(a.excite)
    env = SO101Env(actuators=None)
    start = env.get_actuators()
    fitted, table = {}, {}
    for j in map(int, a.joints.split(",")):
        p, before, after = fit_joint(env, log, j, start)
        fitted[JOINTS[j]] = p
        table[JOINTS[j]] = {"rmse_default_deg": round(before, 2), "rmse_fitted_deg": round(after, 2),
                            "default": {k: round(start[JOINTS[j]][k], 4) for k in PARAMS},
                            "fitted": {k: round(v, 4) for k, v in p.items()}}
        print(f"{JOINTS[j]:13s} excitation RMSE {before:6.2f} -> {after:6.2f} deg   "
              + " ".join(f"{k}={v:.4g}" for k, v in p.items()), flush=True)
    report = {"excitation": table, "dead_band_deg": dead_band(log), "servo_registers": regs}
    if a.validate:
        env_fit = SO101Env(actuators=fitted)
        report["validation_rmse_deg"] = validate(SO101Env(actuators=None), env_fit, a.validate, a.repo_id)
        print("held-out teleop RMSE (deg):", json.dumps(report["validation_rmse_deg"]))
    Path(a.out).write_text(json.dumps(fitted, indent=1))
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1))
    print("dead band (deg):", json.dumps(report["dead_band_deg"]))
    print(f"wrote {a.out} and {a.report}")


if __name__ == "__main__":
    main()
