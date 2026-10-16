"""Object-centric demonstration generation (a simplified MimicGen).

Adapts one recorded demonstration to a new object pose:

  1. split it at the grasp and the release (from the gripper command);
  2. compute the grasp-centre path and approach direction by forward kinematics;
  3. shift the path by the object's displacement: ramped in while the gripper is
     still high (finished before the final descent), held through the grasp,
     ramped out before the release (the box does not move); turn the wrist by
     the object's change in yaw;
  4. solve IK along the new path, keeping the operator's approach direction and
     gripper timing.

Callers keep a generated trajectory only if it succeeds in physics.
"""

from __future__ import annotations

import numpy as np

from .kinematics import SO101IK
from .sim import FPS


def wrap90(a):
    """Wrap an angle to [-45, 45) deg: the jaws grip a square face equally at a and a + 90 deg."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def segment(actions):
    """(grasp, release) frame indices. The grasp is the first close *after* the
    gripper opened for the approach: real demonstrations start and end with the
    gripper shut at rest."""
    g = actions[:, 5]
    closed = g < (np.percentile(g, 5) + np.percentile(g, 95)) / 2
    opened = np.where(~closed)[0]
    if not len(opened):
        raise ValueError("gripper never opens in this demonstration")
    shut = np.where(closed[opened[0]:])[0]
    if not len(shut):
        raise ValueError("gripper never closes in this demonstration")
    t_grasp = int(opened[0] + shut[0])
    after = np.where(~closed[t_grasp:])[0]
    return t_grasp, t_grasp + int(after[0]) if len(after) else len(g) - 1


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def descent_start(heights, t_grasp, clearance=0.10):
    """Last frame before the grasp with the grasp centre still `clearance` above the table."""
    above = np.where(heights[:t_grasp] > clearance)[0]
    return int(above[-1]) if len(above) else max(0, t_grasp - int(0.8 * FPS))


def displacement_weight(T, t_grasp, t_release, t_descent, ramp_in=1.2, ramp_out=1.0):
    """Share of the object's displacement applied at each frame: 0 -> 1 ending at
    t_descent, 1 while holding, 1 -> 0 before the release."""
    t = np.arange(T)
    a1 = max(1, t_descent)
    a0 = max(0, a1 - int(ramp_in * FPS))
    w_in = _smoothstep((t - a0) / max(1, a1 - a0))
    r0 = t_grasp + int(0.25 * FPS)                                   # lift clear first
    r1 = min(max(r0 + 1, t_release - int(0.15 * FPS)), r0 + int(ramp_out * FPS * 2))
    w_out = 1 - _smoothstep((t - r0) / max(1, r1 - r0))
    return np.where(t <= t_grasp, w_in, w_out)


def generate(env, ik: SO101IK, actions, src_xy, src_yaw, new_xy, new_yaw, shift_timing="before"):
    """Joint commands (T x 6, LeRobot units) doing the demonstrated task on the moved object.

    shift_timing="before" completes the displacement before the final descent, so
    the gripper comes straight down onto the object; "during" ramps it over the
    last ramp_in seconds before the grasp (the naive schedule, kept as an ablation)."""
    actions = np.asarray(actions, float)
    t_grasp, t_release = segment(actions)
    q_src = env.lerobot_to_rad(actions)
    points = np.array([ik.grasp_point(q[:5]) for q in q_src])
    axes = np.array([ik.fk(q[:5])[1] for q in q_src])
    t_descent = descent_start(points[:, 2], t_grasp) if shift_timing == "before" else t_grasp
    w = displacement_weight(len(actions), t_grasp, t_release, t_descent)
    shift = np.r_[np.asarray(new_xy) - np.asarray(src_xy), 0.0]
    src_bearing, new_bearing = np.arctan2(src_xy[1], src_xy[0]), np.arctan2(new_xy[1], new_xy[0])
    d_roll = wrap90((new_yaw - new_bearing) - (src_yaw - src_bearing))   # jaws relative to the object
    out = np.empty_like(actions)
    q = q_src[0, :5].copy()
    for t in range(len(actions)):
        c, s = np.cos(w[t] * (new_bearing - src_bearing)), np.sin(w[t] * (new_bearing - src_bearing))
        axis = axes[t].copy()                                        # operator's approach, re-aimed
        axis[:2] = [c * axis[0] - s * axis[1], s * axis[0] + c * axis[1]]
        q, _ = ik.solve(points[t] + w[t] * shift, approach=axis, seed=q, roll=q_src[t, 4] + w[t] * d_roll)
        out[t, :5] = env.rad_to_lerobot(np.r_[q, 0])[:5]
        out[t, 5] = actions[t, 5]                                    # gripper timing unchanged
    return out
