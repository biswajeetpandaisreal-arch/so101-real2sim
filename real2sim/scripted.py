"""Scripted pick-and-place demonstrations (test scaffolding, not training data).

Generates demonstrations in the same format as real `lerobot-record` episodes,
so the replay / generation / training pipeline can be built and debugged before
the real recordings exist. Waypoints are expressed for the grasp centre and
converted to joint targets with SO101IK; small timing and path perturbations
make them less perfectly regular.
"""

from __future__ import annotations

import numpy as np

from .kinematics import SO101IK
from .sim_env import FPS

GRIP_OPEN, GRIP_CLOSED = 60.0, 0.0
READY = np.array([0.20, 0.0, 0.16])          # grasp-centre "ready" point (m)
CARRY_Z = 0.13


def wrap90(a):
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def _min_jerk(n):
    s = np.linspace(0, 1, n + 1)[1:]
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def plan(env, ik: SO101IK, cube_xy, cube_yaw=0.0, rng=None, perturb=True):
    """Joint-target trajectory (T x 6, LeRobot units) for one pick-and-place."""
    rng = rng or np.random.default_rng()
    cx, cy = cube_xy
    bx, by = env.basket_pos[:2]
    roll = wrap90(cube_yaw - np.arctan2(cy, cx))
    k = rng.uniform(0.85, 1.15) if perturb else 1.0
    jit = (lambda: rng.normal(0, 0.004, 3)) if perturb else (lambda: np.zeros(3))
    # (target grasp-centre point, duration s, gripper at end, roll)
    segs = [
        (np.array([cx, cy, 0.12]) + jit(), 1.6, GRIP_OPEN, roll),
        (np.array([cx, cy, 0.03]), 1.0, GRIP_OPEN, roll),
        (None, 0.6, GRIP_CLOSED, roll),                       # close
        (np.array([cx, cy, CARRY_Z]), 0.8, GRIP_CLOSED, roll),
        (np.array([bx, by, CARRY_Z]) + jit(), 1.6, GRIP_CLOSED, 0.0),
        (np.array([bx, by, 0.10]), 0.5, GRIP_CLOSED, 0.0),
        (None, 0.5, GRIP_OPEN, 0.0),                          # release
        (READY + jit(), 1.4, GRIP_OPEN, 0.0),
    ]
    q, _ = ik.solve(READY, seed=np.deg2rad([0, 0, 0, 90, 0]))
    grip = GRIP_OPEN
    p_now = READY.copy()
    out = [np.r_[env.rad_to_lerobot(np.r_[q, 0])[:5], grip]]
    for target, dur, g_end, r_end in segs:
        n = max(2, int(round(dur * k * FPS)))
        s = _min_jerk(n)
        r0 = q[4]
        g0 = grip
        for si in s:
            r = r0 + si * (r_end - r0)
            if target is not None:
                p = p_now + si * (target - p_now)
                q, _ = ik.solve(p, seed=q, roll=r)
            else:
                q = q.copy()
                q[4] = r
            g = g0 + si * (g_end - g0)
            out.append(np.r_[env.rad_to_lerobot(np.r_[q, 0])[:5], g])
        if target is not None:
            p_now = target
        grip = g_end
    for _ in range(int(0.5 * FPS)):          # hold at the end
        out.append(out[-1].copy())
    return np.array(out)


def run(env, actions, cube_xy, cube_yaw=0.0, writer=None):
    """Execute a joint-target trajectory in physics; returns success."""
    env.reset(cube_xy, cube_yaw, q0_lerobot=actions[0])
    for a in actions:
        if writer is not None:
            writer.add(env, a)
        env.step(a)
    return env.success()
