"""Forward and inverse kinematics of the SO-101 grasp centre.

The arm has five joints, so it cannot reach an arbitrary 6-D pose. IK solves for
the grasp-centre *position* plus, as a soft goal, the *approach direction*
(wrist -> fingertips); wrist roll is set separately to orient the jaws.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares

ARM = 5                                         # shoulder_pan .. wrist_roll (gripper excluded)
EXTRA_SEEDS = np.deg2rad([[0, 0, 0, 90, 0], [0, 30, -30, 80, 0], [0, -20, 40, 70, 0]])


def min_jerk(a, b, s):
    """Minimum-jerk interpolation from a to b, s in [0, 1] (scalar or array)."""
    s = np.clip(s, 0.0, 1.0)
    return a + (b - a) * (10 * s**3 - 15 * s**4 + 6 * s**5)


class SO101IK:
    def __init__(self, env):
        self.env = env
        self.m = env.model
        self.d = mujoco.MjData(env.model)       # private copy: IK never disturbs the simulation
        self.qadr = env.qadr[:ARM]
        self.lo, self.hi = env.range[:ARM, 0], env.range[:ARM, 1]
        self.site = env.model.site("gripperframe").id
        self.wrist = env.model.body("gripper").id
        self.grasp_offset = self._grasp_offset()

    def _grasp_offset(self, half_object=0.02, clearance=0.012):
        """Grasp centre in the gripperframe site's frame.

        The site sits at fingertip height but only ~20 mm from the fixed jaw, so
        centring it on the object lands the fixed jaw on the object's edge. The
        grasp centre is placed half an object width (the tealight's radius is
        19 mm) plus a clearance in from the fixed jaw, along the closing direction."""
        m, d = self.m, self.d
        d.qpos[:] = 0
        d.qpos[self.qadr] = np.deg2rad([0, 0, 0, 90, 0])
        d.qpos[self.env.qadr[5]] = np.deg2rad(60)          # jaw open
        mujoco.mj_kinematics(m, d)
        s = d.site_xpos[self.site].copy()
        R = d.site_xmat[self.site].reshape(3, 3)
        approach = s - d.xpos[self.wrist]
        approach /= np.linalg.norm(approach)
        fixed = np.mean([d.geom_xpos[m.geom(n).id] for n in
                         ("fixed_jaw_sph_tip1", "fixed_jaw_box3", "fixed_jaw_box4")], axis=0)
        moving = np.mean([d.geom_xpos[m.geom(n).id] for n in ("moving_jaw_sph_tip1", "moving_jaw_box2")], axis=0)
        closing = (moving - fixed) - ((moving - fixed) @ approach) * approach
        closing /= np.linalg.norm(closing)
        c = fixed + (half_object + clearance) * closing
        c -= ((c - s) @ approach) * approach               # keep it at fingertip height
        return R.T @ (c - s)

    def fk(self, q):
        """(gripperframe site position, unit approach direction) for 5 arm joint angles (rad)."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        p = self.d.site_xpos[self.site].copy()
        a = p - self.d.xpos[self.wrist]
        return p, a / np.linalg.norm(a)

    def grasp_point(self, q):
        p, _ = self.fk(q)
        return p + self.d.site_xmat[self.site].reshape(3, 3) @ self.grasp_offset

    def solve(self, p_target, approach=(0, 0, -1.0), seed=None, roll=0.0, w_axis=0.03):
        """Arm joint angles (rad) putting the grasp centre at p_target, as close to
        the approach direction as the joint limits allow; returns (q, position error m).

        Bounded least squares (joint limits handled exactly). Position dominates:
        the direction term is weighted w_axis (m per rad of tilt), so the solver
        tilts the gripper rather than missing the target when pointing straight
        down is out of reach, as a teleoperator would. Wrist roll is held fixed."""
        approach = np.asarray(approach, float) / np.linalg.norm(approach)
        seeds = ([np.zeros(ARM)] if seed is None else [np.array(seed[:ARM], float)]) + list(EXTRA_SEEDS)
        best = None
        for q0 in seeds:
            def residual(x):
                p, a = self.fk(np.r_[x, roll])
                p = p + self.d.site_xmat[self.site].reshape(3, 3) @ self.grasp_offset
                return np.r_[p - p_target, w_axis * (a - approach)]

            x0 = np.clip(q0[:4], self.lo[:4] + 1e-6, self.hi[:4] - 1e-6)
            r = least_squares(residual, x0, bounds=(self.lo[:4], self.hi[:4]), xtol=1e-10, ftol=1e-10)
            q = np.r_[r.x, roll]
            p, a = self.fk(q)
            p = p + self.d.site_xmat[self.site].reshape(3, 3) @ self.grasp_offset
            err = float(np.linalg.norm(p - p_target))
            tilt = float(np.degrees(np.arccos(np.clip(a @ approach, -1, 1))))
            cost = err + 1e-4 * tilt
            if best is None or cost < best[0]:
                best = (cost, q, err)
            if seed is not None and err < 5e-4:          # the continuation seed reached it: stay on its branch
                break
        return best[1], best[2]
