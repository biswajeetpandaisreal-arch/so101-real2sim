"""Object-centric demonstration generation (a simplified MimicGen).

Takes one recorded demonstration and a new cube pose and produces a new joint
trajectory that performs the same task on the moved cube:

  1. split the demonstration at the grasp and the release (from the gripper
     command);
  2. compute the grasp-centre path and approach direction with forward
     kinematics;
  3. shift the path by the cube's displacement — ramped in during the approach,
     held through the grasp, ramped out before the release (the basket does not
     move) — and turn the wrist by the cube's change in yaw;
  4. solve IK along the new path, keeping the operator's approach direction and
     gripper timing.

Every generated trajectory must then succeed in physics to be kept.
"""

from __future__ import annotations

import numpy as np

from .kinematics import SO101IK
from .scripted import wrap90
from .sim_env import FPS


def segment(actions, closed_below=None):
    """Indices of the grasp (gripper closes) and release (gripper re-opens)."""
    g = actions[:, 5]
    lo, hi = np.percentile(g, 5), np.percentile(g, 95)
    thr = closed_below if closed_below is not None else (lo + hi) / 2
    closed = g < thr
    if not closed.any():
        raise ValueError("gripper never closes in this demonstration")
    t_grasp = int(np.argmax(closed))
    after = np.where(~closed[t_grasp:])[0]
    t_release = t_grasp + int(after[0]) if len(after) else len(g) - 1
    return t_grasp, t_release


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def descent_start(heights, t_grasp, clearance=0.10):
    """Last moment before the grasp when the grasp centre is still high above the
    table (the final descent begins after it)."""
    above = np.where(heights[:t_grasp] > clearance)[0]
    return int(above[-1]) if len(above) else max(0, t_grasp - int(0.8 * FPS))


def displacement_weight(T, t_grasp, t_release, t_descent, ramp_in=1.2, ramp_out=1.0):
    """0 -> 1 while still high above the table (finished before the final
    descent, so the gripper comes straight down onto the moved cube), 1 while
    holding, 1 -> 0 before the release."""
    t = np.arange(T)
    a1 = max(1, t_descent)
    a0 = max(0, a1 - int(ramp_in * FPS))
    w_in = _smoothstep((t - a0) / max(1, a1 - a0))
    r0 = t_grasp + int(0.25 * FPS)                           # lift clear first
    r1 = max(r0 + 1, t_release - int(0.15 * FPS))
    r1 = min(r1, r0 + int(ramp_out * FPS * 2))
    w_out = 1 - _smoothstep((t - r0) / max(1, r1 - r0))
    return np.where(t <= t_grasp, w_in, w_out)


def generate(env, ik: SO101IK, actions, src_xy, src_yaw, new_xy, new_yaw, shift_timing="before"):
    """New joint-target trajectory (T x 6, LeRobot units) for the moved cube.

    shift_timing="before": the displacement is complete before the final descent
    (the gripper comes straight down onto the moved cube). "during": it ramps in
    over the last ramp_in seconds before the grasp, i.e. during the descent (the
    naive schedule; kept as an ablation)."""
    actions = np.asarray(actions, float)
    T = len(actions)
    t_g, t_r = segment(actions)
    q_src = env.lerobot_to_rad(actions)
    pts, axes = [], []
    for q in q_src:
        p, a = ik.fk(q[:5])
        pts.append(ik.grasp_point(q[:5]))
        axes.append(a)
    pts, axes = np.array(pts), np.array(axes)
    t_d = descent_start(pts[:, 2], t_g) if shift_timing == "before" else t_g
    w = displacement_weight(T, t_g, t_r, t_d)
    d_xy = np.r_[np.asarray(new_xy) - np.asarray(src_xy), 0.0]
    # wrist: rotate the jaws by the cube's yaw change relative to the new bearing
    src_bearing = np.arctan2(src_xy[1], src_xy[0])
    new_bearing = np.arctan2(new_xy[1], new_xy[0])
    d_roll = wrap90((new_yaw - new_bearing) - (src_yaw - src_bearing))
    out = np.empty_like(actions)
    q = q_src[0, :5].copy()
    for t in range(T):
        p = pts[t] + w[t] * d_xy
        roll = q_src[t, 4] + w[t] * d_roll
        # keep the operator's approach direction, re-aimed at the new bearing
        ax = axes[t].copy()
        rot = w[t] * (new_bearing - src_bearing)
        c, s = np.cos(rot), np.sin(rot)
        ax[:2] = [c * ax[0] - s * ax[1], s * ax[0] + c * ax[1]]
        q, _ = ik.solve(p, approach=ax, seed=q, roll=roll)
        out[t, :5] = env.rad_to_lerobot(np.r_[q, 0])[:5]
        out[t, 5] = actions[t, 5]                       # gripper timing unchanged
    return out
