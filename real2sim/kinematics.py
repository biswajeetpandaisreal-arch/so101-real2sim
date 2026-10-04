"""Inverse kinematics for the SO-101 tool centre point (gripperframe site).

The SO-101 has five arm joints, so it cannot reach an arbitrary 6-D pose. We
solve for the TCP position plus the *approach direction* (wrist -> fingertips),
which is what a top-down grasp needs; wrist roll is solved separately to align
the jaws with the cube's faces.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares

ARM = 5                                   # shoulder_pan .. wrist_roll (gripper excluded)


class SO101IK:
    def __init__(self, env):
        self.env = env
        self.m = env.model
        self.d = mujoco.MjData(env.model)       # private copy: IK never disturbs the sim
        self.qadr = env.qadr[:ARM]
        self.dadr = env.dadr[:ARM]
        self.lo, self.hi = env.range[:ARM, 0], env.range[:ARM, 1]
        self.site = env.tcp
        self.wrist = env.model.body("gripper").id
        self.Jp = np.zeros((3, env.model.nv))
        self.Jr = np.zeros((3, env.model.nv))
        self.grasp_offset = self._grasp_offset()

    def _grasp_offset(self, half_object=0.02, clearance=0.012):
        """Grasp centre in the gripperframe site's frame.

        The site sits at fingertip height but only ~20 mm from the fixed jaw, so
        centring it on a 4 cm cube lands the fixed jaw on the cube's edge. The
        grasp centre is placed half an object width plus a clearance in from the
        fixed jaw, along the closing direction, measured from the jaw geoms."""
        m, d = self.m, self.d
        d.qpos[:] = 0
        d.qpos[self.qadr] = np.deg2rad([0, 0, 0, 90, 0])
        d.qpos[self.env.qadr[5]] = np.deg2rad(60)          # jaw open
        mujoco.mj_kinematics(m, d)
        s = d.site_xpos[self.site].copy()
        R = d.site_xmat[self.site].reshape(3, 3)
        app = s - d.xpos[self.wrist]
        app /= np.linalg.norm(app)
        f = np.mean([d.geom_xpos[m.geom(n).id] for n in
                     ("fixed_jaw_sph_tip1", "fixed_jaw_box3", "fixed_jaw_box4")], axis=0)
        mv = np.mean([d.geom_xpos[m.geom(n).id] for n in
                      ("moving_jaw_sph_tip1", "moving_jaw_box2")], axis=0)
        w = (mv - f) - ((mv - f) @ app) * app
        w /= np.linalg.norm(w)
        c = f + (half_object + clearance) * w
        c -= ((c - s) @ app) * app                 # keep it at fingertip height
        return R.T @ (c - s)

    def grasp_point(self, q):
        p, _ = self.fk(q)
        return p + self.d.site_xmat[self.site].reshape(3, 3) @ self.grasp_offset

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        p = self.d.site_xpos[self.site].copy()
        a = p - self.d.xpos[self.wrist]
        return p, a / np.linalg.norm(a)

    def solve(self, p_target, approach=(0, 0, -1.0), seed=None, roll=0.0, w_axis=0.03,
              max_tilt_deg=None):
        """q (5 arm joints, rad) putting the grasp centre at p_target, as close to the
        given approach direction as the joint limits allow.

        Bounded least squares (joint limits handled exactly). Position dominates:
        the direction term is weighted w_axis (metres per radian of tilt), so the
        solver tilts the gripper rather than missing the target when pointing
        straight down is out of range — as a human teleoperator would.
        Wrist roll is fixed (it only turns the jaws about the approach axis).
        Returns (q, position error in m)."""
        approach = np.asarray(approach, float) / np.linalg.norm(approach)
        seeds = [np.zeros(ARM) if seed is None else np.array(seed[:ARM], float)]
        seeds += [np.deg2rad(s) for s in ([0, 0, 0, 90, 0], [0, 30, -30, 80, 0],
                                         [0, -20, 40, 70, 0])]
        best = None
        for q0 in seeds:
            q0 = q0.copy()
            q0[4] = roll

            def residual(x):
                q = np.r_[x, roll]
                p, a = self.fk(q)
                p = p + self.d.site_xmat[self.site].reshape(3, 3) @ self.grasp_offset
                return np.r_[p - p_target, w_axis * (a - approach)]

            x0 = np.clip(q0[:4], self.lo[:4] + 1e-6, self.hi[:4] - 1e-6)
            r = least_squares(residual, x0, bounds=(self.lo[:4], self.hi[:4]),
                              xtol=1e-10, ftol=1e-10)
            q = np.r_[r.x, roll]
            p, a = self.fk(q)
            p = p + self.d.site_xmat[self.site].reshape(3, 3) @ self.grasp_offset
            err = float(np.linalg.norm(p - p_target))
            tilt = float(np.degrees(np.arccos(np.clip(a @ approach, -1, 1))))
            cost = err + 1e-4 * tilt
            if max_tilt_deg is not None and tilt > max_tilt_deg:
                cost += 1.0
            if best is None or cost < best[0]:
                best = (cost, q, err)
            if seed is not None and err < 5e-4:      # continuation found it: stay on branch
                break
        return best[1], best[2]

    def solve_path(self, points, approach=(0, 0, -1.0), seed=None, roll=0.0):
        """Solve a sequence of TCP points, seeding each from the previous
        solution so the arm stays on one smooth branch."""
        qs, errs, q = [], [], seed
        for p in points:
            q, err = self.solve(np.asarray(p, float), approach, seed=q, roll=roll)
            qs.append(q.copy())
            errs.append(err)
        return np.array(qs), np.array(errs)
